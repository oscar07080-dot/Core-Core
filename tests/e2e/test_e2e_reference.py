import pytest

from coreedit.cli import main
from coreedit.reference import detect_cuts, extract_template

pytestmark = pytest.mark.e2e


def test_scene_detection_finds_hard_cuts(reference_video):
    cuts = detect_cuts(reference_video)
    # the synthetic reference has 11 cuts (12 x 0.5s scenes); detection of
    # visually similar adjacent sources may miss a couple
    assert 7 <= len(cuts) <= 13


def test_template_extraction(reference_video):
    template = extract_template(reference_video)
    assert len(template.cut_lengths_in_beats) >= 8
    assert template.source_bpm > 0
    assert all(n > 0 for n in template.cut_lengths_in_beats)


def test_pipeline_with_reference(click_song, sample_clips, reference_video, tmp_path):
    out = str(tmp_path / "ref_edit.mp4")
    rc = main([
        "--song", click_song,
        "--clips-dir", sample_clips,
        "--reference", reference_video,
        "--duration", "6",
        "--seed", "3",
        "--output", out,
    ])
    assert rc == 0
    import os
    assert os.path.getsize(out) > 10_000
