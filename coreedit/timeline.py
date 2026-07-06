"""Cut-list construction: turn a beat grid (+ optional rhythm template) into segments."""

from __future__ import annotations

import math
import random

from .audio import snap_to_grid
from .models import BeatGrid, Clip, RhythmTemplate, SectionOverride, Segment

# fraction of segments cut at onset density (vs beat density) per intensity level
INTENSITY_ENERGY_CUTOFF = {"low": 1.01, "medium": 0.75, "high": 0.5}

MIN_SEGMENT = 0.1  # seconds; never emit cuts shorter than this


def boundaries_from_template(
    template: RhythmTemplate, grid: BeatGrid, target_duration: float
) -> list[float]:
    """Apply a reference rhythm template to a new song's beat grid.

    Rescales the template's beat-unit cut lengths to the new tempo, tiles or
    truncates to fill target_duration, and snaps each boundary to the song's
    nearest beat/onset.
    """
    durations = template.durations_for_bpm(grid.bpm)
    snap_candidates = sorted(set(grid.beat_times) | set(grid.onset_times))
    tolerance = grid.beat_period / 4

    boundaries = [0.0]
    t = 0.0
    i = 0
    while t < target_duration:
        t += durations[i % len(durations)]  # tile the template as needed
        i += 1
        snapped = snap_to_grid(t, snap_candidates, tolerance)
        if snapped - boundaries[-1] >= MIN_SEGMENT:
            boundaries.append(min(snapped, target_duration))
    if target_duration - boundaries[-1] >= MIN_SEGMENT:
        boundaries.append(target_duration)
    else:
        boundaries[-1] = target_duration
    return boundaries


def boundaries_from_heuristic(
    grid: BeatGrid, target_duration: float, intensity: str = "medium"
) -> list[float]:
    """Default pacing without a reference: cut every beat, densify to onsets
    in high-energy sections (per the intensity setting)."""
    cutoff = INTENSITY_ENERGY_CUTOFF.get(intensity, 0.75)
    times: set[float] = {t for t in grid.beat_times if t < target_duration}
    for t in grid.onset_times:
        if t < target_duration and grid.energy_at(t) >= cutoff:
            times.add(t)

    boundaries = [0.0]
    for t in sorted(times):
        if t - boundaries[-1] >= MIN_SEGMENT:
            boundaries.append(t)
    if target_duration - boundaries[-1] >= MIN_SEGMENT:
        boundaries.append(target_duration)
    else:
        boundaries[-1] = target_duration
    return boundaries


def thin_onsets_by_strength(
    times: list[float],
    strengths: list[float],
    rng: random.Random,
    min_keep_prob: float = 0.35,
) -> list[float]:
    """Keep onsets with probability scaled by how pronounced each one is, so
    accented notes/hits reliably get their own cut while quieter ones often
    merge into the previous segment. Without this, cutting on "every onset"
    is a metronomically constant rate; this gives it real musical texture.

    `min_keep_prob` is the keep-probability floor for the weakest onset
    (1.0 = keep everything, uniform; lower = more variation). If no strength
    data is available (mismatched/empty `strengths`), every onset is treated
    as equally strong and always kept.
    """
    if not times:
        return []
    if len(strengths) != len(times):
        strengths = [1.0] * len(times)
    peak = max(strengths) if strengths else 0.0
    kept = [times[0]]  # always keep the first onset in range
    for t, s in zip(times[1:], strengths[1:]):
        rel = (s / peak) if peak > 0 else 1.0
        keep_prob = min_keep_prob + (1 - min_keep_prob) * rel
        if rng.random() < keep_prob:
            kept.append(t)
    return kept


def apply_section_overrides(
    boundaries: list[float],
    grid: BeatGrid,
    overrides: list[SectionOverride],
    seed: int | None = None,
    min_keep_prob: float = 0.35,
) -> list[float]:
    """Within each override's time range, replace the normal cut boundaries
    with the grid's per-note ("harmonic") or per-hit ("percussive") onsets,
    thinned by onset strength so the pacing isn't a constant, metronomic rate.

    Boundaries outside every override range are left untouched.
    """
    if not overrides:
        return boundaries
    target_duration = boundaries[-1]

    clipped = [
        SectionOverride(start=max(0.0, o.start), end=min(o.end, target_duration), kind=o.kind)
        for o in overrides
        if o.start < target_duration and o.end > 0
    ]
    if not clipped:
        return boundaries

    rng = random.Random(seed)
    points = {b for b in boundaries if not any(o.start < b < o.end for o in clipped)}
    for o in clipped:
        if o.kind == "harmonic":
            times, strengths = grid.harmonic_onset_times, grid.harmonic_onset_strength
        else:
            times, strengths = grid.percussive_onset_times, grid.percussive_onset_strength
        if len(strengths) != len(times):
            strengths = [1.0] * len(times)
        in_range = [(t, s) for t, s in zip(times, strengths) if o.start <= t <= o.end]
        range_times = [t for t, _ in in_range]
        range_strengths = [s for _, s in in_range]
        points.update(thin_onsets_by_strength(range_times, range_strengths, rng, min_keep_prob))
        points.add(o.start)
        points.add(o.end)
    points.add(0.0)
    points.add(target_duration)

    result = sorted(points)
    merged = [result[0]]
    for t in result[1:]:
        if t - merged[-1] >= MIN_SEGMENT:
            merged.append(t)
    if merged[-1] < target_duration - 1e-9:
        merged.append(target_duration)
    else:
        merged[-1] = target_duration
    return merged


def build_timeline(
    boundaries: list[float],
    clips: list[Clip],
    source_margin: float = 0.5,
    no_repeat_window: int = 3,
    seed: int | None = None,
) -> list[Segment]:
    """Assign a source clip and in-point to every inter-boundary segment."""
    if len(clips) == 0:
        raise ValueError("no source clips available")
    if len(boundaries) < 2:
        raise ValueError("timeline needs at least one segment")

    rng = random.Random(seed)
    window = min(no_repeat_window, len(clips) - 1)
    recent: list[int] = []  # indices of recently used clips
    cursors: dict[int, float] = {}  # per-clip forward in-point cursor

    segments: list[Segment] = []
    for a, b in zip(boundaries, boundaries[1:]):
        seg_dur = b - a
        candidates = [
            i for i, c in enumerate(clips)
            if i not in recent and _fits(c, seg_dur, source_margin)
        ]
        if not candidates:
            # relax the no-repeat constraint before giving up
            candidates = [i for i, c in enumerate(clips) if _fits(c, seg_dur, source_margin)]
        if not candidates:
            raise ValueError(
                f"no clip long enough for a {seg_dur:.2f}s segment "
                f"(with --source-margin {source_margin}); "
                "add longer clips or reduce the margin"
            )
        idx = rng.choice(candidates)
        clip = clips[idx]

        if clip.is_image:
            in_point = 0.0
        else:
            lo = source_margin
            hi = clip.duration - source_margin - seg_dur
            cursor = cursors.get(idx, lo)
            if cursor > hi:
                cursor = lo  # wrap around when the clip is exhausted
            in_point = rng.uniform(cursor, min(hi, cursor + seg_dur * 4))
            cursors[idx] = in_point + seg_dur

        segments.append(Segment(clip=clip, in_point=in_point, duration=seg_dur))
        recent.append(idx)
        if len(recent) > window:
            recent.pop(0)
    return segments


def _fits(clip: Clip, seg_dur: float, margin: float) -> bool:
    if clip.is_image:
        return True
    return clip.duration - 2 * margin >= seg_dur and math.isfinite(clip.duration)
