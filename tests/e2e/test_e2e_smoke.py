import json
import subprocess

import pytest

from coreedit.audio import analyze_song
from coreedit.cli import main

pytestmark = pytest.mark.e2e


def ffprobe_json(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)


def test_click_track_bpm_detected(click_song):
    grid = analyze_song(click_song)
    # 120 BPM click track; accept the half-tempo octave (60) too
    assert grid.bpm == pytest.approx(120, rel=0.08) or grid.bpm == pytest.approx(60, rel=0.08)
    assert len(grid.onset_times) >= 15


def test_full_pipeline_renders_valid_mp4(click_song, sample_clips, tmp_path):
    out = str(tmp_path / "edit.mp4")
    rc = main([
        "--song", click_song,
        "--clips-dir", sample_clips,
        "--duration", "8",
        "--seed", "1",
        "--output", out,
    ])
    assert rc == 0

    info = ffprobe_json(out)
    streams = {s["codec_type"]: s for s in info["streams"]}
    assert streams["video"]["width"] == 1080
    assert streams["video"]["height"] == 1920
    assert streams["video"]["codec_name"] == "h264"
    assert streams["audio"]["codec_name"] == "aac"
    assert float(info["format"]["duration"]) == pytest.approx(8.0, abs=0.3)


def test_dry_run_prints_cut_list(click_song, sample_clips, capsys):
    rc = main([
        "--song", click_song,
        "--clips-dir", sample_clips,
        "--duration", "6",
        "--seed", "2",
        "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "cut list (dry run):" in out
    assert ".mp4" in out
