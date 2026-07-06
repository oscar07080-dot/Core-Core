"""Cut-list construction: turn a beat grid (+ optional rhythm template) into segments."""

from __future__ import annotations

import math
import random

from .audio import snap_to_grid
from .models import BeatGrid, Clip, RhythmTemplate, SectionOverride, Segment

# fraction of segments cut at onset density (vs beat density) per intensity level
INTENSITY_ENERGY_CUTOFF = {"low": 1.01, "medium": 0.75, "high": 0.5}

MIN_SEGMENT = 0.1  # seconds; never emit cuts shorter than this


def snap_boundaries_to_frame_grid(boundaries: list[float], fps: int) -> list[float]:
    """Snap every boundary to the nearest output-frame timestamp (a multiple
    of 1/fps), computed from each boundary's own absolute time.

    Rendering applies an fps filter per segment (needed so `concat` sees a
    uniform frame rate across segments of different source fps), which
    rounds each segment's duration to a whole number of frames measured from
    that segment's own start. Feeding it segments whose *intended* durations
    already aren't frame-aligned means every segment absorbs a small
    rounding error, and across many short segments (fast cut sections) those
    errors compound into audio/video drift that grows over the section. This
    fixes it at the source: durations derived from already-frame-aligned
    boundaries are exact multiples of a frame, so there's nothing left to
    round away, and no cumulative error can build (each boundary is snapped
    independently, not against the previous one).
    """
    if not boundaries:
        return boundaries
    frame_dur = 1.0 / fps
    snapped = [round(b / frame_dur) * frame_dur for b in boundaries]
    result = [snapped[0]]
    for b in snapped[1:]:
        if b - result[-1] < frame_dur / 2:
            b = result[-1] + frame_dur  # avoid a zero/negative-length segment
        result.append(b)
    return result


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


def select_accent_onsets(
    times: list[float],
    strengths: list[float],
    variation: float = 0.35,
    local_window: float = 2.0,
    max_gap: float | None = None,
) -> list[float]:
    """Deterministically keep the onsets that are accents relative to their
    local neighborhood: an onset gets a cut iff its strength is at least
    `(1 - variation)` of the loudest onset within `local_window` seconds of
    it. `variation=1` keeps every onset (uniform); `variation=0` keeps only
    exact local peaks.

    This replaced a probabilistic version (keep-probability scaled by
    strength): random per-note coin flips meant a repeated riff cut
    differently on each repetition, strong accents sometimes didn't cut at
    all, and runs of weak notes sometimes all cut -- individually beat-aligned
    but rhythmically arbitrary, which reads as "not in tune with the music"
    even when every cut lands on a real note. A deterministic accent
    threshold makes the same musical phrase always produce the same cut
    pattern, with every accent reliably cutting.

    The comparison is against the *local* peak, not the loudest onset across
    the whole range, so a quiet passage's own accents still cut instead of
    being crushed by one loud peak elsewhere in the section.

    `max_gap` (seconds), when set, guarantees no stretch between kept onsets
    exceeds it: the strongest skipped onset inside an oversized gap is
    promoted (repeatedly, until every gap fits). This bounds "not rapid
    enough" stretches without disturbing accent timing elsewhere.

    If strength data is missing/mismatched, every onset is treated as equally
    strong and kept.
    """
    if not times:
        return []
    if len(strengths) != len(times):
        strengths = [1.0] * len(times)
    threshold = 1.0 - variation
    half = local_window / 2

    kept = set()
    for i, (t, s) in enumerate(zip(times, strengths)):
        local_peak = max(
            sj for tj, sj in zip(times, strengths) if t - half <= tj <= t + half
        )
        if local_peak <= 0 or s >= threshold * local_peak:
            kept.add(i)

    if max_gap is not None and kept:
        while True:
            ordered = sorted(kept)
            oversized = None
            for a, b in zip(ordered, ordered[1:]):
                if times[b] - times[a] > max_gap and b - a > 1:
                    oversized = (a, b)
                    break
            if oversized is None:
                break
            a, b = oversized
            best = max(range(a + 1, b), key=lambda j: strengths[j])
            kept.add(best)

    return [times[i] for i in sorted(kept)]


def apply_section_overrides(
    boundaries: list[float],
    grid: BeatGrid,
    overrides: list[SectionOverride],
    variation: float = 0.35,
    drum_variation: float | None = None,
    local_window: float = 2.0,
) -> list[float]:
    """Within each override's time range, replace the normal cut boundaries
    with the grid's per-note ("harmonic") or per-hit ("percussive") onsets,
    keeping the ones that are accents relative to their local neighborhood
    (see select_accent_onsets). Cut timing in these ranges is fully
    deterministic — the same song and settings always produce the same cut
    pattern; --seed only affects which clips fill the segments.

    `drum_variation` lets the drum-hit ("percussive") ranges use a different
    variation/speed than the melodic ("harmonic") ones -- defaults to
    `variation` (same value for both) when not given.

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

    # never let an override stretch go more than ~2 beats without a cut
    max_gap = 2 * grid.beat_period if grid is not None else None

    points = {b for b in boundaries if not any(o.start < b < o.end for o in clipped)}
    for o in clipped:
        if o.kind == "harmonic":
            times, strengths = grid.harmonic_onset_times, grid.harmonic_onset_strength
            var = variation
        else:
            times, strengths = grid.percussive_onset_times, grid.percussive_onset_strength
            var = drum_variation if drum_variation is not None else variation
        if len(strengths) != len(times):
            strengths = [1.0] * len(times)
        in_range = [(t, s) for t, s in zip(times, strengths) if o.start <= t <= o.end]
        range_times = [t for t, _ in in_range]
        range_strengths = [s for _, s in in_range]
        points.update(
            select_accent_onsets(range_times, range_strengths, var, local_window, max_gap)
        )
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
