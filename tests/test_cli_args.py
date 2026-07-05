import pytest

from coreedit.cli import build_parser, default_output


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
