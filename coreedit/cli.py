"""CLI entrypoint and pipeline orchestration."""

from __future__ import annotations

import argparse
import datetime
import os
import re
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="edit.py",
        description=(
            "Generate a TikTok-ready beat-synced montage edit from a song and "
            "source clips, optionally modeling cut pacing on a reference edit."
        ),
    )
    p.add_argument("--song", required=True, help="audio file for the edit's soundtrack")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--urls", help="text file of source URLs (one `URL|START|END` per line)")
    src.add_argument("--clips-dir", help="directory of local video/image files")

    p.add_argument("--reference", help="reference edit video to copy cut pacing from")
    p.add_argument("--scene-threshold", type=float, default=12.0, dest="scene_threshold",
                   help="cut-detection sensitivity for --reference (lower = more sensitive, "
                        "catches more subtle cuts but risks false positives; default 12.0)")
    p.add_argument("--subject", default="edit", help="label used for the output filename")
    p.add_argument("--output", help="output mp4 path (default: {subject}_{timestamp}.mp4)")
    p.add_argument("--duration", type=float, default=20.0, help="edit length in seconds (default 20)")
    p.add_argument("--song-start", type=float, default=0.0, dest="song_start",
                   help="offset into the song to start from (seconds)")
    p.add_argument("--intensity", choices=["low", "medium", "high"], default="medium",
                   help="cut density when no --reference is given")
    p.add_argument("--no-repeat-window", type=int, default=3, dest="no_repeat_window",
                   help="how many recent clips to avoid reusing back-to-back")
    p.add_argument("--source-margin", type=float, default=0.5, dest="source_margin",
                   help="seconds to skip at the start/end of every source clip")
    p.add_argument("--width", type=int, default=1080)
    p.add_argument("--height", type=int, default=1920)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, help="random seed for reproducible clip selection")
    p.add_argument("--cache-dir", default="cache", dest="cache_dir",
                   help="where URL downloads are cached (default ./cache)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the computed cut list and exit without rendering")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def default_output(subject: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", subject.lower()).strip("-") or "edit"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{slug}_{stamp}.mp4"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from . import acquisition, audio, timeline
    from .ffmpeg_utils import check_binaries
    from .models import EditSpec

    check_binaries()

    if not os.path.isfile(args.song):
        print(f"error: song file not found: {args.song}", file=sys.stderr)
        return 1

    # 1. acquire source clips
    if args.urls:
        with open(args.urls) as f:
            entries = acquisition.parse_url_list(f.read())
        if not entries:
            print(f"error: no URLs found in {args.urls}", file=sys.stderr)
            return 1
        print(f"downloading {len(entries)} source(s) via yt-dlp...")
        paths = acquisition.download_urls(entries, args.cache_dir, verbose=args.verbose)
    else:
        paths = acquisition.collect_local_clips(args.clips_dir)
        if not paths:
            print(f"error: no video/image files in {args.clips_dir}", file=sys.stderr)
            return 1
    clips = acquisition.build_clips(paths)
    print(f"{len(clips)} source clip(s) ready")

    # 2. analyze the song
    print("analyzing song rhythm...")
    grid = audio.analyze_song(args.song, start=args.song_start, duration=args.duration)
    duration = min(args.duration, grid.duration)
    print(f"  bpm={grid.bpm:.1f}  beats={len(grid.beat_times)}  onsets={len(grid.onset_times)}")

    # 3. cut boundaries: reference template or default heuristic
    if args.reference:
        from . import reference as ref_mod

        print(f"extracting cut rhythm from reference: {args.reference}")
        template = ref_mod.extract_template(args.reference, scene_threshold=args.scene_threshold)
        print(f"  {len(template.cut_lengths_in_beats)} cuts @ {template.source_bpm:.1f} bpm")
        boundaries = timeline.boundaries_from_template(template, grid, duration)
    else:
        boundaries = timeline.boundaries_from_heuristic(grid, duration, args.intensity)
    print(f"{len(boundaries) - 1} segments over {duration:.1f}s")

    # 4. build the timeline
    segments = timeline.build_timeline(
        boundaries,
        clips,
        source_margin=args.source_margin,
        no_repeat_window=args.no_repeat_window,
        seed=args.seed,
    )

    if args.dry_run:
        print("\ncut list (dry run):")
        t = 0.0
        for n, seg in enumerate(segments):
            name = os.path.basename(seg.clip.path)
            print(f"  {n:3d}  {t:7.2f}s  {seg.duration:5.2f}s  {name}"
                  f"  [in {seg.in_point:.2f}s]{' (image)' if seg.clip.is_image else ''}")
            t += seg.duration
        return 0

    # 5. render
    output = args.output or default_output(args.subject)
    spec = EditSpec(
        song=args.song,
        output=output,
        duration=duration,
        song_start=args.song_start,
        width=args.width,
        height=args.height,
        fps=args.fps,
        intensity=args.intensity,
        no_repeat_window=args.no_repeat_window,
        source_margin=args.source_margin,
        seed=args.seed,
        clips=clips,
    )
    from . import render as render_mod

    print(f"rendering {len(segments)} segments -> {output}")
    render_mod.render(segments, spec, verbose=args.verbose)
    print(f"done: {output}")
    return 0
