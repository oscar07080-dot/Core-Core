"""E2E fixtures: synthesize test media with ffmpeg lavfi (nothing committed to git)."""

import subprocess

import pytest


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        check=True,
    )


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("media")


@pytest.fixture(scope="session")
def click_song(media_dir):
    """A 12s click track at 120 BPM: a short 1kHz burst every 0.5s.

    Gives librosa real onsets so beat tracking has structure to find.
    """
    path = str(media_dir / "song.wav")
    # 1kHz sine gated on for the first 60ms of every 0.5s cycle
    _ffmpeg(
        "-f", "lavfi",
        "-i", "sine=frequency=1000:sample_rate=44100:duration=12",
        "-af", "volume='if(lt(mod(t,0.5),0.06),1,0)':eval=frame",
        path,
    )
    return path


@pytest.fixture(scope="session")
def sample_clips(media_dir):
    """Three distinct short test videos + one PNG still image."""
    clips_dir = media_dir / "clips"
    clips_dir.mkdir()
    for name, src in [
        ("a.mp4", "testsrc=size=640x360:rate=30:duration=6"),
        ("b.mp4", "testsrc2=size=320x568:rate=25:duration=5"),
        ("c.mp4", "smptebars=size=1280x720:rate=30:duration=7"),
    ]:
        _ffmpeg("-f", "lavfi", "-i", src, "-pix_fmt", "yuv420p", str(clips_dir / name))
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=purple:size=800x600:duration=0.1",
        "-frames:v", "1", str(clips_dir / "still.png"),
    )
    return str(clips_dir)


@pytest.fixture(scope="session")
def reference_video(media_dir, click_song):
    """A ~6s video with hard cuts every 0.5s between visually distinct sources,
    with the click track as audio — a synthetic 'reference edit'."""
    path = str(media_dir / "reference.mp4")
    sources = ["testsrc", "testsrc2", "smptebars", "rgbtestsrc"]
    n = 12
    inputs = []
    graph = []
    for i in range(n):
        src = sources[i % len(sources)]
        inputs += ["-f", "lavfi", "-i", f"{src}=size=320x568:rate=30:duration=0.5"]
        graph.append(f"[{i}:v]setpts=PTS-STARTPTS,format=yuv420p[v{i}];")
    labels = "".join(f"[v{i}]" for i in range(n))
    graph.append(f"{labels}concat=n={n}:v=1:a=0[outv]")
    _ffmpeg(
        *inputs,
        "-i", click_song,
        "-filter_complex", "".join(graph),
        "-map", "[outv]", "-map", f"{n}:a",
        "-shortest", "-c:v", "libx264", "-c:a", "aac",
        path,
    )
    return path
