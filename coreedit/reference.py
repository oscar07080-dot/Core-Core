"""Extract a cut-rhythm template from a reference edit video."""

from __future__ import annotations

from .models import RhythmTemplate

MIN_CUT_LENGTH = 0.08  # seconds; merge detections closer than ~2 frames


def detect_cuts(path: str, min_scene_len_frames: int = 2, threshold: float = 12.0) -> list[float]:
    """Detect hard-cut timestamps (seconds) in a video using PySceneDetect.

    Uses ContentDetector rather than AdaptiveDetector: this genre of edit is
    all static-camera hard cuts with no pans/motion, so AdaptiveDetector's
    motion-robustness (its whole reason for existing) just makes it miss
    cuts between visually-similar consecutive clips (e.g. the same subject
    reshot with a minor color change) — exactly the kind of cut this tool
    needs to catch. A lower threshold catches more subtle cuts at the cost
    of occasional false positives on rapid motion; tune with `threshold`.
    """
    from scenedetect import ContentDetector, detect

    scenes = detect(path, ContentDetector(threshold=threshold, min_scene_len=min_scene_len_frames))
    # scene list is [(start, end), ...]; cuts are the boundaries between scenes
    return [float(start.seconds) for start, _ in scenes[1:]]


def cuts_to_template(cut_times: list[float], video_duration: float, bpm: float) -> RhythmTemplate:
    """Convert absolute cut timestamps into a tempo-portable rhythm template."""
    boundaries = [0.0, *sorted(cut_times), video_duration]
    lengths = []
    for a, b in zip(boundaries, boundaries[1:]):
        d = b - a
        if d >= MIN_CUT_LENGTH:
            lengths.append(d)
        elif lengths:
            lengths[-1] += d  # merge sub-frame slivers into the previous cut
    if not lengths:
        raise ValueError("no usable cuts found in reference video")
    period = 60.0 / bpm if bpm > 0 else 0.5
    return RhythmTemplate(
        cut_lengths_in_beats=[d / period for d in lengths],
        source_bpm=bpm,
    )


def extract_template(path: str, scene_threshold: float = 12.0) -> RhythmTemplate:
    """Full reference analysis: scene cuts + the reference's own tempo."""
    from .audio import analyze_song
    from .ffmpeg_utils import media_duration

    cut_times = detect_cuts(path, threshold=scene_threshold)
    duration = media_duration(path)
    grid = analyze_song(path)
    return cuts_to_template(cut_times, duration, grid.bpm)
