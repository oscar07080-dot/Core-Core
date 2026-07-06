import pytest

from coreedit.cli import build_parser, default_output, parse_time_range


def test_urls_and_clips_dir_mutually_exclusive():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--song", "s.mp3", "--urls", "u.txt", "--clips-dir", "d"]
        )


def test_source_required():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--song", "s.mp3"])


def test_song_required():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--clips-dir", "d"])


def test_defaults():
    args = build_parser().parse_args(["--song", "s.mp3", "--clips-dir", "d"])
    assert args.duration == 20.0
    assert args.song_start == 0.0
    assert args.intensity == "medium"
    assert args.no_repeat_window == 3
    assert args.source_margin == 0.5
    assert args.width == 1080 and args.height == 1920 and args.fps == 30


def test_default_output_slug():
    name = default_output("My Cool Subject!!")
    assert name.startswith("my-cool-subject_")
    assert name.endswith(".mp4")


def test_default_output_empty_subject():
    assert default_output("!!!").startswith("edit_")


def test_parse_time_range_valid():
    assert parse_time_range("5:10") == (5.0, 10.0)
    assert parse_time_range("0.5:2.25") == (0.5, 2.25)


def test_parse_time_range_end_before_start():
    with pytest.raises(ValueError, match="END must be greater"):
        parse_time_range("10:5")


def test_parse_time_range_negative_start():
    with pytest.raises(ValueError, match="must not be negative"):
        parse_time_range("-1:5")


def test_parse_time_range_bad_format():
    with pytest.raises(ValueError, match="expected START:END"):
        parse_time_range("5-10")


def test_parse_time_range_non_numeric():
    with pytest.raises(ValueError, match="must be numbers"):
        parse_time_range("a:b")


def test_chorus_and_drum_buildup_flags_repeatable():
    args = build_parser().parse_args([
        "--song", "s.mp3", "--clips-dir", "d",
        "--chorus", "10:20", "--chorus", "40:50",
        "--drum-buildup", "5:9",
    ])
    assert args.chorus == ["10:20", "40:50"]
    assert args.drum_buildup == ["5:9"]
    assert args.auto_chorus is False
