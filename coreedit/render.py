"""Render a segment timeline to a TikTok-ready mp4 with a single ffmpeg run."""

from __future__ import annotations

import os
import tempfile

from .ffmpeg_utils import run_ffmpeg
from .models import EditSpec, Segment


def build_filtergraph(
    segments: list[Segment],
    input_index: dict[str, int],
    width: int,
    height: int,
    fps: int,
) -> str:
    """Build the filter_complex script text for the segment list.

    `input_index` maps each unique source path to its ffmpeg input number.
    The song is expected to be the input AFTER all video inputs; audio is
    mapped directly in build_command, not through the graph.
    """
    lines = []
    scale = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1,fps={fps},format=yuv420p"
    )
    for n, seg in enumerate(segments):
        i = input_index[seg.clip.path]
        if seg.clip.is_image:
            # looped image input: take the first seg.duration seconds
            lines.append(
                f"[{i}:v]trim=start=0:end={seg.duration:.4f},"
                f"setpts=PTS-STARTPTS,{scale}[seg{n}];"
            )
        else:
            lines.append(
                f"[{i}:v]trim=start={seg.in_point:.4f}:end={seg.out_point:.4f},"
                f"setpts=PTS-STARTPTS,{scale}[seg{n}];"
            )
    concat_inputs = "".join(f"[seg{n}]" for n in range(len(segments)))
    lines.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=0[outv]")
    return "\n".join(lines)


def build_command(
    segments: list[Segment],
    spec: EditSpec,
    filtergraph_path: str,
) -> tuple[list[str], dict[str, int]]:
    """Build the full ffmpeg argument list (without the leading binary/-y)."""
    unique_paths: list[str] = []
    for seg in segments:
        if seg.clip.path not in unique_paths:
            unique_paths.append(seg.clip.path)
    input_index = {p: i for i, p in enumerate(unique_paths)}

    args: list[str] = []
    path_is_image = {seg.clip.path: seg.clip.is_image for seg in segments}
    for path in unique_paths:
        if path_is_image[path]:
            # loop long enough to cover the longest segment using this image
            max_dur = max(s.duration for s in segments if s.clip.path == path)
            args += ["-loop", "1", "-t", f"{max_dur + 1:.2f}", "-i", path]
        else:
            args += ["-i", path]

    song_index = len(unique_paths)
    args += ["-ss", f"{spec.song_start:.3f}", "-i", spec.song]

    args += [
        "-filter_complex_script", filtergraph_path,
        "-map", "[outv]",
        "-map", f"{song_index}:a",
        "-t", f"{spec.duration:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", str(spec.fps),
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-shortest",
        spec.output,
    ]
    return args, input_index


def render(segments: list[Segment], spec: EditSpec, verbose: bool = False) -> None:
    """Render the timeline to spec.output."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".filtergraph", delete=False
    ) as f:
        graph_path = f.name
    try:
        args, input_index = build_command(segments, spec, graph_path)
        graph = build_filtergraph(
            segments, input_index, spec.width, spec.height, spec.fps
        )
        with open(graph_path, "w") as f:
            f.write(graph)
        run_ffmpeg(args, verbose=verbose)
    finally:
        os.unlink(graph_path)
