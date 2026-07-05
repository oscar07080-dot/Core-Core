"""Thin wrappers around the ffmpeg/ffprobe system binaries."""

from __future__ import annotations

import json
import shutil
import subprocess


class FfmpegError(RuntimeError):
    pass


def check_binaries() -> None:
    """Fail fast with a clear message if ffmpeg/ffprobe are missing."""
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if missing:
        raise FfmpegError(
            f"required binaries not found on PATH: {', '.join(missing)}. "
            "Install ffmpeg (e.g. `apt install ffmpeg` or `brew install ffmpeg`)."
        )


def probe(path: str) -> dict:
    """Return ffprobe format+streams info for a media file as a dict."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_format", "-show_streams",
        "-of", "json", path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FfmpegError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def media_duration(path: str) -> float:
    """Duration of a media file in seconds."""
    info = probe(path)
    dur = info.get("format", {}).get("duration")
    if dur is None:
        # some containers report duration only on streams
        for stream in info.get("streams", []):
            if "duration" in stream:
                dur = stream["duration"]
                break
    if dur is None:
        raise FfmpegError(f"could not determine duration of {path}")
    return float(dur)


def run_ffmpeg(args: list[str], verbose: bool = False) -> None:
    """Run ffmpeg with the given args, raising on failure."""
    cmd = ["ffmpeg", "-hide_banner", "-y", *args]
    if verbose:
        print("+ " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=not verbose, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        tail = "\n".join(stderr.splitlines()[-15:])
        raise FfmpegError(f"ffmpeg failed (exit {result.returncode}):\n{tail}")
