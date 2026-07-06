import pytest

from coreedit.models import SectionOverride
from coreedit.timeline import apply_section_overrides, boundaries_from_heuristic


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
