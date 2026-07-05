import math

from coreedit.models import Clip, EditSpec, Segment
from coreedit.render import build_command, build_filtergraph


def make_segments():
    a = Clip(path="/fake/a.mp4", duration=10.0)
    b = Clip(path="/fake/b.mp4", duration=8.0)
    img = Clip(path="/fake/c.png", duration=math.inf, is_image=True)
    return [
        Segment(clip=a, in_point=1.0, duration=0.5),
        Segment(clip=b, in_point=2.0, duration=0.25),
        Segment(clip=a, in_point=4.0, duration=0.5),  # reuses input 0
        Segment(clip=img, in_point=0.0, duration=0.4),
    ]


def spec(out="/tmp/out.mp4"):
    return EditSpec(song="/fake/song.mp3", output=out, duration=2.0)


def test_unique_inputs_opened_once():
    segments = make_segments()
    args, input_index = build_command(segments, spec(), "/tmp/graph.txt")
    assert len(input_index) == 3  # a, b, c — not 4
    assert args.count("-i") == 4  # 3 sources + 1 song


def test_song_is_last_input_and_mapped_as_audio():
    segments = make_segments()
    args, input_index = build_command(segments, spec(), "/tmp/graph.txt")
    song_index = len(input_index)
    assert f"{song_index}:a" in args
    assert "[outv]" in args


def test_image_input_gets_loop_flag():
    segments = make_segments()
    args, _ = build_command(segments, spec(), "/tmp/graph.txt")
    i = args.index("/fake/c.png")
    assert args[i - 1] == "-i"
    assert "-loop" in args[: i]


def test_filtergraph_labels_and_concat():
    segments = make_segments()
    _, input_index = build_command(segments, spec(), "/tmp/graph.txt")
    graph = build_filtergraph(segments, input_index, 1080, 1920, 30)
    for n in range(len(segments)):
        assert f"[seg{n}]" in graph
    assert f"concat=n={len(segments)}:v=1:a=0[outv]" in graph
    # brackets must balance
    assert graph.count("[") == graph.count("]")


def test_filtergraph_trim_points():
    segments = make_segments()
    _, input_index = build_command(segments, spec(), "/tmp/graph.txt")
    graph = build_filtergraph(segments, input_index, 1080, 1920, 30)
    assert "trim=start=1.0000:end=1.5000" in graph
    assert "trim=start=2.0000:end=2.2500" in graph
    # image segment trims from 0
    assert "trim=start=0:end=0.4000" in graph


def test_filtergraph_scales_to_canvas():
    segments = make_segments()
    _, input_index = build_command(segments, spec(), "/tmp/graph.txt")
    graph = build_filtergraph(segments, input_index, 720, 1280, 24)
    assert "scale=720:1280:force_original_aspect_ratio=increase" in graph
    assert "crop=720:1280" in graph
    assert "fps=24" in graph
