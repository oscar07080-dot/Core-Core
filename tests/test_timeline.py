import math

import pytest

from coreedit.models import Clip
from coreedit.timeline import (
    boundaries_from_heuristic,
    build_timeline,
    snap_boundaries_to_frame_grid,
)


def test_heuristic_low_intensity_cuts_on_beats_only(steady_grid):
    boundaries = boundaries_from_heuristic(steady_grid, 10.0, intensity="low")
    # every interior boundary should be a beat time (0.5s multiples)
    for b in boundaries[1:-1]:
        assert b % 0.5 == pytest.approx(0.0, abs=1e-6)


def test_heuristic_high_intensity_denser_than_low(steady_grid):
    low = boundaries_from_heuristic(steady_grid, 20.0, intensity="low")
    high = boundaries_from_heuristic(steady_grid, 20.0, intensity="high")
    assert len(high) > len(low)


def test_heuristic_ends_at_target(steady_grid):
    boundaries = boundaries_from_heuristic(steady_grid, 7.3, intensity="medium")
    assert boundaries[0] == 0.0
    assert boundaries[-1] == pytest.approx(7.3)


def test_build_timeline_durations_sum_to_target(steady_grid, clip_pool):
    boundaries = boundaries_from_heuristic(steady_grid, 10.0)
    segments = build_timeline(boundaries, clip_pool, seed=42)
    assert sum(s.duration for s in segments) == pytest.approx(10.0)


def test_build_timeline_deterministic_with_seed(steady_grid, clip_pool):
    boundaries = boundaries_from_heuristic(steady_grid, 10.0)
    a = build_timeline(boundaries, clip_pool, seed=7)
    b = build_timeline(boundaries, clip_pool, seed=7)
    assert [(s.clip.path, s.in_point, s.duration) for s in a] == [
        (s.clip.path, s.in_point, s.duration) for s in b
    ]


def test_no_immediate_repeats(steady_grid, clip_pool):
    boundaries = boundaries_from_heuristic(steady_grid, 15.0)
    segments = build_timeline(boundaries, clip_pool, no_repeat_window=1, seed=1)
    for prev, cur in zip(segments, segments[1:]):
        assert prev.clip.path != cur.clip.path


def test_in_points_stay_within_clip(steady_grid, clip_pool):
    boundaries = boundaries_from_heuristic(steady_grid, 15.0)
    margin = 0.5
    segments = build_timeline(boundaries, clip_pool, source_margin=margin, seed=3)
    for s in segments:
        if s.clip.is_image:
            assert s.in_point == 0.0
        else:
            assert s.in_point >= margin - 1e-9
            assert s.out_point <= s.clip.duration - margin + 1e-9


def test_single_clip_pool_allows_repeats(steady_grid):
    clips = [Clip(path="/fake/only.mp4", duration=30.0)]
    boundaries = boundaries_from_heuristic(steady_grid, 5.0)
    segments = build_timeline(boundaries, clips, seed=1)
    assert all(s.clip.path == "/fake/only.mp4" for s in segments)


def test_segment_too_long_for_all_clips_raises(steady_grid):
    clips = [Clip(path="/fake/tiny.mp4", duration=0.6)]
    with pytest.raises(ValueError, match="no clip long enough"):
        build_timeline([0.0, 5.0], clips, source_margin=0.5)


def test_image_fits_any_duration():
    clips = [Clip(path="/fake/pic.png", duration=math.inf, is_image=True)]
    segments = build_timeline([0.0, 60.0], clips, seed=1)
    assert segments[0].duration == 60.0


def test_empty_clips_raises():
    with pytest.raises(ValueError, match="no source clips"):
        build_timeline([0.0, 1.0], [])


# --- snap_boundaries_to_frame_grid: fixes drift compounding across many segments ---


def test_snap_rounds_to_nearest_frame():
    fps = 30
    frame = 1 / fps
    boundaries = [0.0, 0.882, 1.486, 8.15]
    snapped = snap_boundaries_to_frame_grid(boundaries, fps)
    for b in snapped:
        # every value must be (very nearly) an exact multiple of the frame duration
        assert abs(round(b / frame) - b / frame) < 1e-6


def test_snap_error_does_not_compound_across_many_boundaries():
    # simulate the real bug: hundreds of short, non-frame-aligned segments
    fps = 30
    boundaries = [0.0]
    t = 0.0
    for _ in range(300):
        t += 0.137  # deliberately not a multiple of 1/30
        boundaries.append(t)

    snapped = snap_boundaries_to_frame_grid(boundaries, fps)
    # each boundary is snapped from its OWN absolute time, so error stays
    # bounded by half a frame regardless of how many boundaries precede it,
    # instead of growing with the count (which is what per-segment rounding
    # against a locally-reset clock did)
    frame = 1 / fps
    for original, snapped_val in zip(boundaries, snapped):
        assert abs(snapped_val - original) <= frame / 2 + 1e-9


def test_snap_keeps_boundaries_strictly_increasing():
    fps = 30
    # two boundaries close enough to collide onto the same frame after snapping
    boundaries = [0.0, 0.01, 0.02, 1.0]
    snapped = snap_boundaries_to_frame_grid(boundaries, fps)
    assert snapped == sorted(snapped)
    assert len(snapped) == len(set(snapped))


def test_snap_preserves_zero_start():
    snapped = snap_boundaries_to_frame_grid([0.0, 1.234, 5.0], 30)
    assert snapped[0] == 0.0


def test_snap_empty_returns_empty():
    assert snap_boundaries_to_frame_grid([], 30) == []
