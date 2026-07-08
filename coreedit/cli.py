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
    p.add_argument("--auto-chorus", action="store_true", dest="auto_chorus",
                   help="auto-detect the chorus (sustained high-energy section) and cut on "
                        "every melodic note there, plus every drum hit in the build-up right "
                        "before it. Ignored where --chorus/--drum-buildup are given explicitly.")
    p.add_argument("--chorus", action="append", default=[], metavar="START:END",
                   help="time range (seconds) to cut on every melodic/guitar note instead of "
                        "the normal pacing; overrides --auto-chorus (repeatable)")
    p.add_argument("--drum-buildup", action="append", default=[], dest="drum_buildup",
                   metavar="START:END",
                   help="time range (seconds) to cut on every drum hit instead of the normal "
                        "pacing (repeatable)")
    p.add_argument("--note-sensitivity", type=float, default=0.02, dest="note_sensitivity",
                   help="how easily a melodic note counts as an onset for --chorus/--auto-chorus "
                        "(lower = catches more/quieter notes but risks false triggers on sustain "
                        "or vibrato; default 0.02)")
    p.add_argument("--note-min-spacing", type=float, default=0.1, dest="note_min_spacing",
                   help="minimum seconds between two detected melodic notes (lower allows faster "
                        "playing to register as separate notes, but risks a single sustained/"
                        "vibrato note being split into several false ones; default 0.1)")
    p.add_argument("--section-variation", type=float, default=0.35, dest="section_variation",
                   help="within --chorus/--auto-chorus ranges, how much accented notes are "
                        "favored over quiet ones (0 = only accents get their own cut, weak ones "
                        "merge into the previous shot; 1 = uniform, every single note cuts "
                        "regardless of accent; default 0.35)")
    p.add_argument("--drum-variation", type=float, default=None, dest="drum_variation",
                   help="same as --section-variation but for --drum-buildup ranges specifically "
                        "(1 = cut on every drum hit, fastest; lower = only accented hits cut); "
                        "defaults to --section-variation's value if not given")
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
    p.add_argument("--export-clips", metavar="DIR", dest="export_clips",
                   help="instead of one rendered mp4, export each segment as its own "
                        "numbered clip (001.mp4, 002.mp4, ...) plus the trimmed song audio "
                        "(song.m4a) into DIR. For reassembling the exact same cut timing in "
                        "an external editor (e.g. CapCut): drag the numbered clips into a "
                        "timeline in order, add song.m4a as the audio track, then use that "
                        "editor's 'replace clip' feature on each one to swap in your own "
                        "footage without disturbing the timing")
    p.add_argument("--export-capcut", nargs="?", const="", metavar="NAME",
                   dest="export_capcut", default=None,
                   help="instead of one rendered mp4, create a native CapCut desktop draft "
                        "(project) named NAME directly in CapCut's drafts folder, with the "
                        "full timeline pre-built: one pre-trimmed clip per cut plus the song "
                        "on the audio track. Open CapCut and the project is on the home "
                        "screen; use Replace on each clip to swap in your own footage "
                        "without touching the timing. Best-effort: the draft format is "
                        "undocumented, so if your CapCut version rejects it, fall back to "
                        "--export-clips. NAME defaults to the output filename's stem")
    p.add_argument("--capcut-drafts-dir", dest="capcut_drafts_dir", metavar="DIR",
                   help="where CapCut keeps its drafts, for --export-capcut (auto-detected "
                        "in the standard Windows/macOS locations when omitted)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def default_output(subject: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", subject.lower()).strip("-") or "edit"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{slug}_{stamp}.mp4"


def parse_time_range(spec: str) -> tuple[float, float]:
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"expected START:END, got {spec!r}")
    try:
        start, end = float(parts[0]), float(parts[1])
    except ValueError:
        raise ValueError(f"START/END must be numbers (seconds): {spec!r}") from None
    if end <= start:
        raise ValueError(f"END must be greater than START: {spec!r}")
    if start < 0:
        raise ValueError(f"START must not be negative: {spec!r}")
    return start, end


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from . import acquisition, audio, timeline
    from .ffmpeg_utils import check_binaries
    from .models import EditSpec, SectionOverride

    check_binaries()

    if not os.path.isfile(args.song):
        print(f"error: song file not found: {args.song}", file=sys.stderr)
        return 1

    try:
        overrides = [
            SectionOverride(start=s, end=e, kind="harmonic")
            for s, e in (parse_time_range(spec) for spec in args.chorus)
        ] + [
            SectionOverride(start=s, end=e, kind="percussive")
            for s, e in (parse_time_range(spec) for spec in args.drum_buildup)
        ]
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
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
    backend = "madmom (neural)" if audio._madmom() else "librosa"
    print(f"analyzing song rhythm... [{backend}]")
    grid = audio.analyze_song(
        args.song, start=args.song_start, duration=args.duration,
        note_sensitivity=args.note_sensitivity, note_min_spacing=args.note_min_spacing,
    )
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

    # 3b. chorus / drum-buildup overrides: explicit ranges win, else auto-detect
    if not overrides and args.auto_chorus:
        chorus, buildup = audio.detect_chorus_and_buildup(grid)
        if chorus:
            overrides.append(chorus)
            print(f"auto-detected chorus: {chorus.start:.1f}s-{chorus.end:.1f}s "
                  "(cutting on every note)")
        else:
            print("auto-chorus: no clear chorus section found, using normal pacing")
        if buildup:
            overrides.append(buildup)
            print(f"auto-detected build-up: {buildup.start:.1f}s-{buildup.end:.1f}s "
                  "(cutting on every drum hit)")
    if overrides:
        boundaries = timeline.apply_section_overrides(
            boundaries, grid, overrides,
            variation=args.section_variation,
            drum_variation=args.drum_variation,
        )
    # align cuts to the output frame grid so per-segment fps conversion at
    # render time can't round each one, which would otherwise compound into
    # audio/video drift across a run of many short segments
    boundaries = timeline.snap_boundaries_to_frame_grid(boundaries, args.fps)
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

    if args.export_capcut is not None:
        from . import capcut

        drafts_dir = args.capcut_drafts_dir or capcut.find_drafts_dir()
        if not drafts_dir:
            print(
                "error: couldn't find CapCut's drafts folder in the standard "
                "locations:\n  "
                + "\n  ".join(capcut._candidate_drafts_dirs())
                + "\npass it explicitly with --capcut-drafts-dir "
                "(in CapCut: Settings -> Drafts to see where it is)",
                file=sys.stderr,
            )
            return 1
        draft_name = args.export_capcut or os.path.splitext(os.path.basename(output))[0]
        print(f"building CapCut draft '{draft_name}' in {drafts_dir}")
        try:
            draft_dir = capcut.export_capcut_draft(
                segments, spec, drafts_dir, draft_name, verbose=args.verbose
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except FileExistsError:
            print(f"error: a draft named '{draft_name}' already exists -- pick "
                  "another name: --export-capcut NAME", file=sys.stderr)
            return 1
        print(
            f"done: {draft_dir}\n"
            "Open CapCut (restart it if it was already running) -- the project "
            f"'{draft_name}' is on the home screen with the timeline pre-built. "
            "Click a clip, hit Replace, pick your own footage; repeat per clip. "
            "The cut timing is locked in by each clip's duration.\n"
            "If CapCut won't open the draft (format changes between versions), "
            "use --export-clips as the fallback."
        )
        return 0

    if args.export_clips:
        print(f"exporting {len(segments)} numbered clips + song audio -> {args.export_clips}")
        render_mod.render_clip_sequence(segments, spec, args.export_clips, verbose=args.verbose)
        print(
            f"done: {args.export_clips}\n"
            "Drag the numbered clips into a timeline in order, add song.m4a as the audio "
            "track, then use your editor's 'replace clip' feature on each one to swap in "
            "your own footage -- the timing stays exactly as computed."
        )
        return 0

    print(f"rendering {len(segments)} segments -> {output}")
    render_mod.render(segments, spec, verbose=args.verbose)
    print(f"done: {output}")
    return 0
