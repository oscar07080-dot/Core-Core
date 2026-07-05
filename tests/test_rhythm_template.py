import pytest

from coreedit.models import RhythmTemplate
from coreedit.reference import cuts_to_template
from coreedit.timeline import boundaries_from_template


def test_cuts_to_template_beat_math():
    # cuts at 1s and 2s in a 3s video at 120 bpm (0.5s beat period)
    template = cuts_to_template([1.0, 2.0], video_duration=3.0, bpm=120.0)
    assert template.cut_lengths_in_beats == pytest.approx([2.0, 2.0, 2.0])
    assert template.source_bpm == 120.0


def test_cuts_merge_sub_frame_slivers():
    # 0.02s gap is below MIN_CUT_LENGTH and merges into the previous cut
    template = cuts_to_template([1.0, 1.02], video_duration=2.0, bpm=60.0)
    assert len(template.cut_lengths_in_beats) == 2
    assert sum(template.cut_lengths_in_beats) == pytest.approx(2.0)


def test_no_cuts_raises():
    with pytest.raises(ValueError):
        cuts_to_template([], video_duration=0.01, bpm=120.0)


def test_durations_rescale_to_new_bpm():
    t = RhythmTemplate(cut_lengths_in_beats=[1.0, 2.0], source_bpm=120.0)
    assert t.durations_for_bpm(60.0) == pytest.approx([1.0, 2.0])
    assert t.durations_for_bpm(120.0) == pytest.approx([0.5, 1.0])


def test_template_tiles_to_fill_duration(steady_grid):
    # one-beat cuts at 120 bpm = 0.5s cuts; a 2-cut template must tile across 10s
    t = RhythmTemplate(cut_lengths_in_beats=[1.0, 1.0], source_bpm=100.0)
    boundaries = boundaries_from_template(t, steady_grid, target_duration=10.0)
    assert boundaries[0] == 0.0
    assert boundaries[-1] == pytest.approx(10.0)
    assert len(boundaries) > 10  # tiled well beyond the template's 2 cuts


def test_template_truncates_to_duration(steady_grid):
    t = RhythmTemplate(cut_lengths_in_beats=[1.0] * 100, source_bpm=120.0)
    boundaries = boundaries_from_template(t, steady_grid, target_duration=3.0)
    assert boundaries[-1] == pytest.approx(3.0)
    assert all(b <= 3.0 for b in boundaries)


def test_template_boundaries_snap_to_grid(steady_grid):
    # slightly-off template lengths should snap onto the 0.25s onset grid
    t = RhythmTemplate(cut_lengths_in_beats=[1.05, 0.95], source_bpm=120.0)
    boundaries = boundaries_from_template(t, steady_grid, target_duration=6.0)
    for b in boundaries[1:-1]:
        assert b % 0.25 == pytest.approx(0.0, abs=1e-6)
