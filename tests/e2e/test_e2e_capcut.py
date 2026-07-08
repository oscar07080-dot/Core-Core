"""E2E: build a real CapCut draft from synthesized media and check its JSON."""

import json
import math
import os

import pytest

pyjyd = pytest.importorskip("pyJianYingDraft")

from coreedit.models import Clip, EditSpec, Segment  # noqa: E402
from coreedit.capcut import export_capcut_draft  # noqa: E402

pytestmark = pytest.mark.e2e

SEC = 1_000_000


@pytest.fixture
def segments(sample_clips):
    paths = sorted(
        os.path.join(sample_clips, f) for f in os.listdir(sample_clips)
        if f.endswith(".mp4")
    )
    a, b, c = (Clip(path=p, duration=5.0) for p in paths)
    img_path = os.path.join(sample_clips, "still.png")
    img = Clip(path=img_path, duration=math.inf, is_image=True)
    return [
        Segment(clip=a, in_point=0.5, duration=0.5),
        Segment(clip=b, in_point=1.0, duration=0.25),
        Segment(clip=img, in_point=0.0, duration=0.3),
        Segment(clip=c, in_point=2.0, duration=0.45),
    ]


def test_draft_structure(tmp_path, segments, click_song):
    spec = EditSpec(
        song=click_song, output=str(tmp_path / "unused.mp4"),
        duration=1.5, song_start=0.0, width=1080, height=1920, fps=30,
    )
    drafts_dir = str(tmp_path / "drafts")
    os.makedirs(drafts_dir)

    draft_dir = export_capcut_draft(segments, spec, drafts_dir, "core_edit")

    assert draft_dir == os.path.join(drafts_dir, "core_edit")
    assert os.path.isfile(os.path.join(draft_dir, "draft_content.json"))
    assert os.path.isfile(os.path.join(draft_dir, "draft_meta_info.json"))

    with open(os.path.join(draft_dir, "draft_content.json")) as f:
        content = json.load(f)

    tracks = {t["type"]: t for t in content["tracks"]}
    assert set(tracks) == {"video", "audio"}
    video_segs = tracks["video"]["segments"]
    assert len(video_segs) == len(segments)
    assert len(tracks["audio"]["segments"]) == 1

    # segments must be back-to-back with durations matching the timeline
    t = 0
    for seg, js in zip(segments, video_segs):
        tr = js["target_timerange"]
        assert tr["start"] == t
        assert tr["duration"] == int(round(seg.duration * SEC))
        # source range must lie within the pre-trimmed clip
        src = js["source_timerange"]
        assert src["start"] == 0
        assert src["duration"] <= tr["duration"]
        t += tr["duration"]
    assert content["duration"] == t

    # all referenced media must live inside the draft folder (self-contained)
    for mat in content["materials"]["videos"] + content["materials"]["audios"]:
        assert mat["path"].startswith(draft_dir)
        assert os.path.isfile(mat["path"])

    # canvas matches the spec
    assert content["canvas_config"]["width"] == 1080
    assert content["canvas_config"]["height"] == 1920
    assert content["fps"] == 30


def test_existing_draft_name_raises(tmp_path, segments, click_song):
    spec = EditSpec(
        song=click_song, output=str(tmp_path / "unused.mp4"),
        duration=1.5, song_start=0.0,
    )
    drafts_dir = str(tmp_path / "drafts")
    os.makedirs(os.path.join(drafts_dir, "taken"))
    with pytest.raises(FileExistsError):
        export_capcut_draft(segments, spec, drafts_dir, "taken")
