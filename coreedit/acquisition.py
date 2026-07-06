"""Source clip acquisition: yt-dlp URL downloads and local directory globbing."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass

from .ffmpeg_utils import media_duration

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class UrlEntry:
    url: str
    start: float | None = None
    end: float | None = None


def parse_url_list(text: str) -> list[UrlEntry]:
    """Parse a URL list file: one `URL|START|END` per line, `#` comments.

    START and END are optional trim points in seconds within the source video.
    """
    entries: list[UrlEntry] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) > 3:
            raise ValueError(f"line {lineno}: too many fields (expected URL|START|END): {line!r}")
        url = parts[0]
        if not url.lower().startswith(("http://", "https://")):
            raise ValueError(f"line {lineno}: not a URL: {url!r}")
        start = end = None
        try:
            if len(parts) >= 2 and parts[1]:
                start = float(parts[1])
            if len(parts) == 3 and parts[2]:
                end = float(parts[2])
        except ValueError:
            raise ValueError(f"line {lineno}: START/END must be numbers (seconds): {line!r}") from None
        if start is not None and end is not None and end <= start:
            raise ValueError(f"line {lineno}: END must be greater than START: {line!r}")
        entries.append(UrlEntry(url=url, start=start, end=end))
    return entries


def cache_key(entry: UrlEntry) -> str:
    """Stable filename stem for a URL entry, including its trim range."""
    raw = f"{entry.url}|{entry.start}|{entry.end}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _detect_proxy() -> str | None:
    """Read the standard proxy env vars.

    yt-dlp's own network requests already honor these automatically, but a
    trimmed (--start/--end) download shells out to ffmpeg as an external
    downloader, and ffmpeg does *not* read HTTPS_PROXY on its own -- it needs
    the proxy passed to it explicitly. Without this, section downloads can
    silently bypass a required proxy and get blocked by the destination
    server instead of actually reaching it through the approved egress path.
    """
    for var in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        value = os.environ.get(var)
        if value:
            return value
    return None


def ydl_options(entry: UrlEntry, cache_dir: str) -> dict:
    """Build the yt-dlp options dict for one entry (kept separate for testing)."""
    opts: dict = {
        "outtmpl": os.path.join(cache_dir, cache_key(entry) + ".%(ext)s"),
        "format": "bv*[height<=1920]+ba/b[height<=1920]/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    proxy = _detect_proxy()
    if proxy:
        opts["proxy"] = proxy
    if entry.start is not None or entry.end is not None:
        from yt_dlp.utils import download_range_func

        start = entry.start or 0.0
        end = entry.end if entry.end is not None else math.inf
        opts["download_ranges"] = download_range_func(None, [(start, end)])
        # section downloads need keyframe-accurate re-encode-free cutting
        opts["force_keyframes_at_cuts"] = True
    return opts


def _find_cached(cache_dir: str, key: str) -> str | None:
    if not os.path.isdir(cache_dir):
        return None
    for name in os.listdir(cache_dir):
        stem, ext = os.path.splitext(name)
        if stem == key and ext.lower() in VIDEO_EXTS | IMAGE_EXTS:
            return os.path.join(cache_dir, name)
    return None


def download_urls(entries: list[UrlEntry], cache_dir: str, verbose: bool = False) -> list[str]:
    """Download each URL entry via yt-dlp (skipping cached ones); return file paths."""
    import yt_dlp

    os.makedirs(cache_dir, exist_ok=True)
    paths: list[str] = []
    for entry in entries:
        key = cache_key(entry)
        cached = _find_cached(cache_dir, key)
        if cached:
            if verbose:
                print(f"cached: {entry.url} -> {cached}")
            paths.append(cached)
            continue
        opts = ydl_options(entry, cache_dir)
        if verbose:
            opts["quiet"] = False
            print(f"downloading: {entry.url}")
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([entry.url])
        downloaded = _find_cached(cache_dir, key)
        if downloaded is None:
            raise RuntimeError(f"yt-dlp reported success but no file found for {entry.url}")
        paths.append(downloaded)
    return paths


def collect_local_clips(clips_dir: str) -> list[str]:
    """List video and image files in a directory (non-recursive), sorted by name."""
    if not os.path.isdir(clips_dir):
        raise NotADirectoryError(f"clips directory not found: {clips_dir}")
    paths = []
    for name in sorted(os.listdir(clips_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext in VIDEO_EXTS | IMAGE_EXTS:
            paths.append(os.path.join(clips_dir, name))
    return paths


def build_clips(paths: list[str]) -> list["Clip"]:
    """Probe each file and wrap it in a Clip model."""
    from .models import Clip

    clips = []
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        if ext in IMAGE_EXTS:
            clips.append(Clip(path=path, duration=math.inf, is_image=True))
        else:
            clips.append(Clip(path=path, duration=media_duration(path), is_image=False))
    return clips
