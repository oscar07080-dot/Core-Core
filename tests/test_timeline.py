import math

import pytest

from coreedit.models import Clip
from coreedit.timeline import (
    boundaries_from_heuristic,
    build_timeline,
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
