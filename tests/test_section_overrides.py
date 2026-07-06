import random

import pytest

from coreedit.models import BeatGrid, SectionOverride
from coreedit.timeline import apply_section_overrides, boundaries_from_heuristic, thin_onsets_by_strength


def test_harmonic_override_uses_harmonic_onsets(steady_grid):
    base = boundaries_from_heuristic(steady_grid, 20.0, intensity="low")
    override = SectionOverride(start=5.0, end=8.0, kind="harmonic")
    result = apply_section_overrides(base, steady_grid, [override])

    inside = [b for b in result if 5.0 < b < 8.0]
    expected = [t for t in steady_grid.harmonic_onset_times if 5.0 <= t <= 8.0]
    assert inside == expected


def test_percussive_override_uses_percussive_onsets(steady_grid):
    base = boundaries_from_heuristic(steady_grid, 20.0, intensity="low")
    override = SectionOverride(start=5.0, end=8.0, kind="percussive")
    result = apply_section_overrides(base, steady_grid, [override])

    inside = [b for b in result if 5.0 < b < 8.0]
    expected = [t for t in steady_grid.percussive_onset_times if 5.0 <= t <= 8.0]
    assert inside == expected


def test_boundaries_outside_override_untouched(steady_grid):
    base = boundaries_from_heuristic(steady_grid, 20.0, intensity="low")
    override = SectionOverride(start=10.0, end=12.0, kind="harmonic")
    result = apply_section_overrides(base, steady_grid, [override])

    before = [b for b in base if b <= 10.0]
    after_result = [b for b in result if b <= 10.0]
    assert before == after_result


def test_no_overrides_returns_original():
    boundaries = [0.0, 1.0, 2.0, 3.0]
    assert apply_section_overrides(boundaries, None, []) == boundaries


def test_override_clipped_to_target_duration(steady_grid):
    base = boundaries_from_heuristic(steady_grid, 10.0, intensity="low")
    override = SectionOverride(start=8.0, end=50.0, kind="harmonic")  # extends past duration
    result = apply_section_overrides(base, steady_grid, [override])
    assert result[-1] == pytest.approx(10.0)
    assert all(b <= 10.0 for b in result)


def test_override_starts_at_zero(steady_grid):
    base = boundaries_from_heuristic(steady_grid, 10.0, intensity="low")
    override = SectionOverride(start=0.0, end=3.0, kind="percussive")
    result = apply_section_overrides(base, steady_grid, [override])
    assert result[0] == 0.0
    inside = [b for b in result if 0.0 < b < 3.0]
    expected = {t for t in steady_grid.percussive_onset_times if 0.0 <= t <= 3.0}
    # onsets landing within MIN_SEGMENT of the 0.0/3.0 edges get merged into
    # the edge itself rather than kept as a separate point
    assert set(inside) <= expected
    assert len(inside) >= len(expected) - 2


def test_multiple_non_overlapping_overrides(steady_grid):
    base = boundaries_from_heuristic(steady_grid, 20.0, intensity="low")
    overrides = [
        SectionOverride(start=2.0, end=4.0, kind="harmonic"),
        SectionOverride(start=12.0, end=14.0, kind="percussive"),
    ]
    result = apply_section_overrides(base, steady_grid, overrides)

    harmonic_zone = [b for b in result if 2.0 < b < 4.0]
    percussive_zone = [b for b in result if 12.0 < b < 14.0]
    expected_harmonic = {t for t in steady_grid.harmonic_onset_times if 2.0 <= t <= 4.0}
    expected_percussive = {t for t in steady_grid.percussive_onset_times if 12.0 <= t <= 14.0}
    # onsets landing within MIN_SEGMENT of a range edge get merged into the edge
    assert set(harmonic_zone) <= expected_harmonic
    assert len(harmonic_zone) >= len(expected_harmonic) - 2
    assert set(percussive_zone) <= expected_percussive
    assert len(percussive_zone) >= len(expected_percussive) - 2


def test_result_stays_monotonic_and_reaches_duration(steady_grid):
    base = boundaries_from_heuristic(steady_grid, 20.0, intensity="low")
    override = SectionOverride(start=1.0, end=19.5, kind="percussive")
    result = apply_section_overrides(base, steady_grid, [override])
    assert result == sorted(result)
    assert len(result) == len(set(result))
    assert result[-1] == pytest.approx(20.0)


# --- thin_onsets_by_strength: the fix for "chorus shouldn't be a constant speed" ---


def test_thin_onsets_no_strength_data_keeps_everything():
    times = [1.0, 2.0, 3.0, 4.0]
    kept = thin_onsets_by_strength(times, [], random.Random(0), min_keep_prob=0.0)
    assert kept == times


def test_thin_onsets_zero_strength_never_kept_at_min_prob_zero():
    times = [1.0, 2.0, 3.0, 4.0]
    strengths = [5.0, 0.0, 5.0, 0.0]  # bimodal: accented / silent
    kept = thin_onsets_by_strength(times, strengths, random.Random(0), min_keep_prob=0.0)
    assert kept == [1.0, 3.0]  # first onset always kept; zero-strength ones never are


def test_thin_onsets_min_keep_prob_one_keeps_everything_regardless_of_strength():
    times = [1.0, 2.0, 3.0]
    strengths = [1.0, 0.001, 5.0]
    kept = thin_onsets_by_strength(times, strengths, random.Random(0), min_keep_prob=1.0)
    assert kept == times


def test_thin_onsets_deterministic_with_same_seed():
    times = list(range(30))
    strengths = [(i % 5) for i in range(30)]
    a = thin_onsets_by_strength(times, strengths, random.Random(42), min_keep_prob=0.3)
    b = thin_onsets_by_strength(times, strengths, random.Random(42), min_keep_prob=0.3)
    assert a == b


def test_thin_onsets_lower_min_keep_prob_thins_more_on_average():
    times = list(range(200))
    strengths = [(i % 4) for i in range(200)]  # mix of strong/weak onsets
    uniform = thin_onsets_by_strength(times, strengths, random.Random(1), min_keep_prob=1.0)
    varied = thin_onsets_by_strength(times, strengths, random.Random(1), min_keep_prob=0.1)
    assert len(varied) < len(uniform)


def _grid_with_strengths() -> BeatGrid:
    times = [round(i * 0.2, 4) for i in range(50)]  # 0.0 .. 9.8, every 0.2s
    strengths = [5.0 if i % 4 == 0 else 0.2 for i in range(50)]  # accented every 4th note
    return BeatGrid(
        bpm=120.0, beat_times=[], onset_times=[],
        energy_times=[0.0, 10.0], energy=[1.0, 1.0], duration=10.0,
        harmonic_onset_times=times, harmonic_onset_strength=strengths,
        percussive_onset_times=times, percussive_onset_strength=strengths,
    )


def test_apply_section_overrides_thins_chorus_with_real_strength_data():
    grid = _grid_with_strengths()
    base = [0.0, 10.0]
    override = SectionOverride(start=0.0, end=10.0, kind="harmonic")

    uniform = apply_section_overrides(base, grid, [override], seed=1, min_keep_prob=1.0)
    varied = apply_section_overrides(base, grid, [override], seed=1, min_keep_prob=0.1)

    # uniform mode cuts on literally every note (50 onsets -> ~51 boundaries)
    assert len(uniform) >= 45
    # varied mode drops most of the quiet in-between notes, keeping mostly accents
    assert len(varied) < len(uniform) * 0.6


# --- local-window normalization: fixes "some parts too rapid, some not rapid enough" ---


def test_thin_onsets_local_window_treats_a_quiet_passage_fairly():
    """A quiet passage's own accents shouldn't be crushed just because one
    loud passage elsewhere in the range has a much higher peak -- each
    passage's cut rate should reflect its own dynamics."""
    # quiet passage 0-5s: accents at strength 1.0 every 4th note, rest 0.04
    # loud passage 10-15s: accents at strength 10.0 every 4th note, rest 0.4
    # (same 10x ratio locally, wildly different absolute scale)
    times = [round(i * 0.2, 4) for i in range(25)] + [round(10 + i * 0.2, 4) for i in range(25)]
    strengths = [1.0 if i % 4 == 0 else 0.04 for i in range(25)] + \
                [10.0 if i % 4 == 0 else 0.4 for i in range(25)]

    global_peak_kept = thin_onsets_by_strength(
        times, strengths, random.Random(3), min_keep_prob=0.1, local_window=1000.0,
    )
    local_kept = thin_onsets_by_strength(
        times, strengths, random.Random(3), min_keep_prob=0.1, local_window=2.0,
    )

    quiet_global = [t for t in global_peak_kept if t < 5.0]
    quiet_local = [t for t in local_kept if t < 5.0]
    # with a global peak (set by the loud passage), the quiet passage's own
    # accents (strength 1.0 vs the loud passage's 10.0) look weak and get
    # thinned much harder than with local comparison
    assert len(quiet_local) > len(quiet_global)


def test_thin_onsets_local_window_keeps_both_passages_proportional():
    times = [round(i * 0.2, 4) for i in range(25)] + [round(10 + i * 0.2, 4) for i in range(25)]
    strengths = [1.0 if i % 4 == 0 else 0.04 for i in range(25)] + \
                [10.0 if i % 4 == 0 else 0.4 for i in range(25)]

    kept = thin_onsets_by_strength(times, strengths, random.Random(5), min_keep_prob=0.05, local_window=2.0)
    quiet_kept = [t for t in kept if t < 5.0]
    loud_kept = [t for t in kept if t >= 10.0]
    # roughly comparable retention rate in both passages (each ~6-7 accents
    # out of 25 onsets), not one passage almost entirely dropped
    assert len(quiet_kept) > 0
    assert abs(len(quiet_kept) - len(loud_kept)) <= 4


# --- separate percussive variation (--drum-variation): speeds up drums independently ---


def test_percussive_min_keep_prob_overrides_harmonic_variation():
    grid = _grid_with_strengths()
    base = [0.0, 10.0]
    overrides = [SectionOverride(start=0.0, end=10.0, kind="percussive")]

    harmonic_setting_ignored = apply_section_overrides(
        base, grid, overrides, seed=2, min_keep_prob=0.9, percussive_min_keep_prob=0.05,
    )
    # percussive_min_keep_prob (0.05, aggressive thinning) should govern here,
    # not min_keep_prob (0.9, near-uniform) which only applies to harmonic
    uniform_percussive = apply_section_overrides(
        base, grid, overrides, seed=2, min_keep_prob=0.9, percussive_min_keep_prob=0.9,
    )
    assert len(harmonic_setting_ignored) < len(uniform_percussive)


def test_percussive_min_keep_prob_defaults_to_min_keep_prob():
    grid = _grid_with_strengths()
    base = [0.0, 10.0]
    overrides = [SectionOverride(start=0.0, end=10.0, kind="percussive")]

    explicit = apply_section_overrides(base, grid, overrides, seed=4, min_keep_prob=0.2)
    same_explicit = apply_section_overrides(
        base, grid, overrides, seed=4, min_keep_prob=0.2, percussive_min_keep_prob=0.2,
    )
    assert explicit == same_explicit
