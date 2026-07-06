import pytest

from coreedit.models import BeatGrid, SectionOverride
from coreedit.timeline import apply_section_overrides, boundaries_from_heuristic, select_accent_onsets


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


# --- select_accent_onsets: deterministic accent selection ("in tune", not random) ---


def test_select_no_strength_data_keeps_everything():
    times = [1.0, 2.0, 3.0, 4.0]
    kept = select_accent_onsets(times, [], variation=0.0)
    assert kept == times


def test_select_zero_strength_never_kept_at_variation_zero():
    times = [1.0, 2.0, 3.0, 4.0]
    strengths = [5.0, 0.0, 5.0, 0.0]  # bimodal: accented / silent
    kept = select_accent_onsets(times, strengths, variation=0.0)
    # 2.0 (strength 0) is the launch note of the rise into the 3.0 peak, so
    # it's force-kept by the rising-run rule despite failing the plain
    # threshold on its own; 4.0 has no follow-on rise, so it's dropped
    assert kept == [1.0, 2.0, 3.0]


def test_select_variation_one_keeps_everything_regardless_of_strength():
    times = [1.0, 2.0, 3.0]
    strengths = [1.0, 0.001, 5.0]
    kept = select_accent_onsets(times, strengths, variation=1.0)
    assert kept == times


def test_select_is_deterministic():
    times = [i * 0.25 for i in range(30)]
    strengths = [(i % 5) + 0.1 for i in range(30)]
    a = select_accent_onsets(times, strengths, variation=0.3)
    b = select_accent_onsets(times, strengths, variation=0.3)
    assert a == b


def test_select_repeated_riff_cuts_identically_each_repetition():
    """The core 'in tune' property the probabilistic version lacked: the same
    musical phrase must produce the same cut pattern every time it repeats."""
    riff_strengths = [5.0, 0.5, 2.0, 0.5]  # one bar: accent, weak, medium, weak
    # one extra trailing bar so every *compared* repetition has a following
    # note to potentially rise into; the true final note in the sequence has
    # no "next" note by definition and is excluded from the comparison below
    n_bars = 5
    times = [i * 0.25 for i in range(4 * n_bars)]
    strengths = riff_strengths * n_bars
    kept = select_accent_onsets(times, strengths, variation=0.35, local_window=1.0)

    kept_set = set(kept)
    pattern_per_bar = [
        tuple((i * 0.25 + bar * 1.0) in kept_set for i in range(4))
        for bar in range(n_bars - 1)
    ]
    assert len(set(pattern_per_bar)) == 1  # every comparable repetition cuts identically
    assert all(p[0] for p in pattern_per_bar)  # the strong beat always cuts
    # and every bar's accent (the 5.0 note) must cut
    assert all(p[0] for p in pattern_per_bar)


def test_select_lower_variation_thins_more():
    times = [i * 0.25 for i in range(200)]
    strengths = [(i % 4) + 0.2 for i in range(200)]  # mix of strong/weak onsets
    uniform = select_accent_onsets(times, strengths, variation=1.0)
    varied = select_accent_onsets(times, strengths, variation=0.1)
    assert len(varied) < len(uniform)


def test_select_max_gap_promotes_strongest_skipped_onset():
    # a loud peak, a long flat weak stretch, then a rise into a final peak
    # (the flat middle has no rise of its own, isolating max_gap's effect
    # from the separate "first note of a rise" rule below)
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    strengths = [10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0]
    no_fill = select_accent_onsets(times, strengths, variation=0.1, local_window=100.0)
    # index 6 is kept too: it's the first (only) step of the rise into the final peak
    assert no_fill == [0.0, 6.0, 7.0]
    filled = select_accent_onsets(
        times, strengths, variation=0.1, local_window=100.0, max_gap=2.0
    )
    for a, b in zip(filled, filled[1:]):
        assert b - a <= 2.0 + 1e-9


# --- first note of a rising run: fixes missing a phrase's quiet launch point ---


def test_select_keeps_quiet_launch_note_of_a_rising_phrase():
    """A phrase that climbs to a loud peak is typically voiced quietest at
    its start -- a pure loudness threshold drops exactly the note that
    marks where the rise begins."""
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    # background beat, then a rise: quiet launch note -> louder -> loud peak
    strengths = [5.0, 0.3, 2.0, 4.0, 9.0]
    kept = select_accent_onsets(times, strengths, variation=0.2, local_window=100.0)
    assert 0.25 in kept  # the quiet launch note, which fails the plain threshold alone
    without_rule = [
        t for t, s in zip(times, strengths)
        if max(strengths) > 0 and s >= 0.8 * max(strengths)
    ]
    assert 0.25 not in without_rule  # confirms the threshold alone would have dropped it


def test_select_rising_run_only_flags_the_launch_note_not_every_step():
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    strengths = [1.0, 2.0, 3.0, 4.0, 5.0]  # one continuous rise, no plateau
    kept = select_accent_onsets(times, strengths, variation=0.0, local_window=100.0)
    # only the launch note (index 0) is force-kept by the rising-run rule;
    # the peak (index 4) is kept via the ordinary loudness threshold
    assert kept == [0.0, 4.0]


def test_select_purely_decreasing_strengths_has_no_forced_launch_notes():
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    strengths = [9.0, 4.0, 3.0, 2.0, 1.0]
    kept = select_accent_onsets(times, strengths, variation=0.0, local_window=100.0)
    assert kept == [0.0]  # only the actual peak, no spurious rise-starts


def test_select_repeated_measure_keeps_the_launch_note_every_time():
    """Mirrors the reported bug: a rising phrase repeats every measure, and
    its quiet launch note must be kept on every repetition, not just some."""
    measure = [5.0, 0.4, 2.5, 8.0]  # beat, quiet launch, rising, peak
    n_measures = 4
    times = [i * 0.25 for i in range(4 * n_measures)]
    strengths = measure * n_measures

    kept = select_accent_onsets(times, strengths, variation=0.2, local_window=1.0)
    kept_set = set(kept)
    for m in range(n_measures):
        launch_time = m * 1.0 + 0.25
        assert launch_time in kept_set, f"measure {m}'s launch note was dropped"


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

    uniform = apply_section_overrides(base, grid, [override], variation=1.0)
    varied = apply_section_overrides(base, grid, [override], variation=0.1)

    # uniform mode cuts on literally every note (50 onsets -> ~51 boundaries)
    assert len(uniform) >= 45
    # varied mode drops most of the quiet in-between notes, keeping mostly accents
    assert len(varied) < len(uniform) * 0.6


def test_apply_section_overrides_is_deterministic():
    grid = _grid_with_strengths()
    base = [0.0, 10.0]
    override = SectionOverride(start=0.0, end=10.0, kind="harmonic")
    a = apply_section_overrides(base, grid, [override], variation=0.35)
    b = apply_section_overrides(base, grid, [override], variation=0.35)
    assert a == b


# --- local-window normalization: fixes "some parts too rapid, some not rapid enough" ---


def test_select_local_window_treats_a_quiet_passage_fairly():
    """A quiet passage's own accents shouldn't be crushed just because one
    loud passage elsewhere in the range has a much higher peak -- each
    passage's cut rate should reflect its own dynamics."""
    # quiet passage 0-5s: accents at strength 1.0 every 4th note, rest 0.04
    # loud passage 10-15s: accents at strength 10.0 every 4th note, rest 0.4
    # (same relative accent ratio locally, wildly different absolute scale)
    times = [round(i * 0.2, 4) for i in range(25)] + [round(10 + i * 0.2, 4) for i in range(25)]
    strengths = [1.0 if i % 4 == 0 else 0.04 for i in range(25)] + \
                [10.0 if i % 4 == 0 else 0.4 for i in range(25)]

    global_peak_kept = select_accent_onsets(times, strengths, variation=0.5, local_window=1000.0)
    local_kept = select_accent_onsets(times, strengths, variation=0.5, local_window=2.0)

    quiet_global = [t for t in global_peak_kept if t < 5.0]
    quiet_local = [t for t in local_kept if t < 5.0]
    # with a global peak (set by the loud passage), the quiet passage's own
    # accents (strength 1.0 vs the loud passage's 10.0) fall below the
    # threshold entirely; with local comparison they cut like accents should
    assert len(quiet_local) > len(quiet_global)


def test_select_local_window_keeps_both_passages_proportional():
    times = [round(i * 0.2, 4) for i in range(25)] + [round(10 + i * 0.2, 4) for i in range(25)]
    strengths = [1.0 if i % 4 == 0 else 0.04 for i in range(25)] + \
                [10.0 if i % 4 == 0 else 0.4 for i in range(25)]

    kept = select_accent_onsets(times, strengths, variation=0.5, local_window=2.0)
    quiet_kept = [t for t in kept if t < 5.0]
    loud_kept = [t for t in kept if t >= 10.0]
    # both passages have accents every 4th note; deterministic local
    # comparison must keep exactly the accents in each, so equal counts
    assert len(quiet_kept) > 0
    assert len(quiet_kept) == len(loud_kept)


# --- separate percussive variation (--drum-variation): speeds up drums independently ---


def test_drum_variation_overrides_harmonic_variation():
    grid = _grid_with_strengths()
    base = [0.0, 10.0]
    overrides = [SectionOverride(start=0.0, end=10.0, kind="percussive")]

    harmonic_setting_ignored = apply_section_overrides(
        base, grid, overrides, variation=1.0, drum_variation=0.05,
    )
    # drum_variation (0.05, accents only) should govern here, not
    # variation (1.0, keep-everything) which only applies to harmonic ranges
    uniform_percussive = apply_section_overrides(
        base, grid, overrides, variation=1.0, drum_variation=1.0,
    )
    assert len(harmonic_setting_ignored) < len(uniform_percussive)


def test_drum_variation_defaults_to_variation():
    grid = _grid_with_strengths()
    base = [0.0, 10.0]
    overrides = [SectionOverride(start=0.0, end=10.0, kind="percussive")]

    implicit = apply_section_overrides(base, grid, overrides, variation=0.2)
    explicit = apply_section_overrides(base, grid, overrides, variation=0.2, drum_variation=0.2)
    assert implicit == explicit
