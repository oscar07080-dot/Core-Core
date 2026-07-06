import math
import os

from coreedit.models import Clip, EditSpec, Segment
from coreedit.render import render_clip_sequence


def make_segments():
    a = Clip(path="/fake/a.mp4", duration=10.0)
    img = Clip(path="/fake/b.png", duration=math.inf, is_image=True)
    return [
        Segment(clip=a, in_point=1.0, duration=0.5),
        Segment(clip=img, in_point=0.0, duration=0.3),
        Segment(clip=a, in_point=4.0, duration=0.75),
    ]


def spec(tmp_path):
    return EditSpec(
        song="/fake/song.mp3", output=str(tmp_path / "unused.mp4"),
        duration=1.55, song_start=2.0,
    )


def test_writes_one_numbered_clip_per_segment(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("coreedit.render.run_ffmpeg", lambda args, verbose=False: calls.append(args))

    out_dir = str(tmp_path / "clips")
    paths = render_clip_sequence(make_segments(), spec(tmp_path), out_dir, verbose=False)

    assert paths == [
        os.path.join(out_dir, "001.mp4"),
        os.path.join(out_dir, "002.mp4"),
        os.path.join(out_dir, "003.mp4"),
    ]
    # one ffmpeg call per segment plus one for the song
    assert len(calls) == 4


def test_video_segment_uses_trim_filter(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("coreedit.render.run_ffmpeg", lambda args, verbose=False: calls.append(args))
    render_clip_sequence(make_segments(), spec(tmp_path), str(tmp_path / "clips"), verbose=False)

    first_call = calls[0]
    assert "/fake/a.mp4" in first_call
    filt = first_call[first_call.index("-filter:v") + 1]
    assert "trim=start=1.0000:end=1.5000" in filt
    assert "-an" in first_call


def test_image_segment_uses_loop_and_duration(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("coreedit.render.run_ffmpeg", lambda args, verbose=False: calls.append(args))
    render_clip_sequence(make_segments(), spec(tmp_path), str(tmp_path / "clips"), verbose=False)

    image_call = calls[1]
    assert "-loop" in image_call
    i = image_call.index("-loop")
    assert image_call[i + 1] == "1"
    assert "-t" in image_call
    assert "0.3000" in image_call[image_call.index("-t") + 1]


def test_no_audio_track_on_exported_clips(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("coreedit.render.run_ffmpeg", lambda args, verbose=False: calls.append(args))
    render_clip_sequence(make_segments(), spec(tmp_path), str(tmp_path / "clips"), verbose=False)
    for call in calls[:3]:  # the 3 clip-export calls, not the trailing song export
        assert "-an" in call


def test_song_exported_trimmed_to_song_start_and_duration(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("coreedit.render.run_ffmpeg", lambda args, verbose=False: calls.append(args))
    render_clip_sequence(make_segments(), spec(tmp_path), str(tmp_path / "clips"), verbose=False)

    song_call = calls[-1]
    assert "/fake/song.mp3" in song_call
    assert "song.m4a" in song_call[-1]
    assert "-ss" in song_call and "2.000" in song_call[song_call.index("-ss") + 1]
    assert "-t" in song_call and "1.550" in song_call[song_call.index("-t") + 1]
    assert "-vn" in song_call


def test_padding_width_scales_with_segment_count(tmp_path, monkeypatch):
    monkeypatch.setattr("coreedit.render.run_ffmpeg", lambda args, verbose=False: None)
    clip = Clip(path="/fake/a.mp4", duration=10000.0)
    many_segments = [Segment(clip=clip, in_point=float(i), duration=0.5) for i in range(1234)]
    paths = render_clip_sequence(many_segments, spec(tmp_path), str(tmp_path / "clips"), verbose=False)
    assert os.path.basename(paths[0]) == "0001.mp4"
    assert os.path.basename(paths[-1]) == "1234.mp4"


def test_output_dir_created(tmp_path, monkeypatch):
    monkeypatch.setattr("coreedit.render.run_ffmpeg", lambda args, verbose=False: None)
    out_dir = str(tmp_path / "nested" / "clips")
    render_clip_sequence(make_segments(), spec(tmp_path), out_dir, verbose=False)
    assert os.path.isdir(out_dir)
