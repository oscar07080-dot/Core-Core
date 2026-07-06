import pytest

from coreedit.audio import analyze_song, detect_chorus_and_buildup
from coreedit.cli import main

pytestmark = pytest.mark.e2e


def test_auto_chorus_finds_the_loud_middle_section(verse_chorus_song):
    grid = analyze_song(verse_chorus_song)
    chorus, buildup = detect_chorus_and_buildup(grid)
    assert chorus is not None
    # the loud section is 6-13s; allow slack for smoothing/threshold edges
    assert 4.0 <= chorus.start <= 8.0
    assert 11.0 <= chorus.end <= 15.0
    assert chorus.kind == "harmonic"
    assert buildup is not None
    assert buildup.end == pytest.approx(chorus.start)
    assert buildup.kind == "percussive"


def test_cli_auto_chorus_runs_end_to_end(verse_chorus_song, sample_clips, tmp_path, capsys):
    out = str(tmp_path / "auto_chorus.mp4")
    rc = main([
        "--song", verse_chorus_song,
        "--clips-dir", sample_clips,
        "--auto-chorus",
        "--duration", "18",
        "--seed", "1",
        "--output", out,
    ])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "auto-detected chorus" in printed


def test_cli_manual_chorus_and_buildup_flags(verse_chorus_song, sample_clips, tmp_path):
    out = str(tmp_path / "manual_sections.mp4")
    rc = main([
        "--song", verse_chorus_song,
        "--clips-dir", sample_clips,
        "--chorus", "6:13",
        "--drum-buildup", "3:6",
        "--duration", "18",
        "--seed", "2",
        "--output", out,
    ])
    assert rc == 0
    import os
    assert os.path.getsize(out) > 10_000


def test_manual_flags_take_precedence_over_auto(verse_chorus_song, sample_clips, tmp_path, capsys):
    out = str(tmp_path / "manual_wins.mp4")
    rc = main([
        "--song", verse_chorus_song,
        "--clips-dir", sample_clips,
        "--auto-chorus",
        "--chorus", "6:13",
        "--duration", "18",
        "--seed", "1",
        "--output", out,
    ])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "auto-detected" not in printed
