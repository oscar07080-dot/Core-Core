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
def verse_chorus_song(media_dir):
    """An 18s song with a clear structure: quiet verse (0-6s, sparse clicks),
    a loud sustained chorus (6-13s, dense guitar-note-like tones over a
    driving click track), then a quiet outro (13-18s) — for testing
    chorus/build-up detection against a real energy contour."""
    path = str(media_dir / "verse_chorus.wav")
    # verse: quiet 1kHz clicks every 0.5s
    verse = "sine=frequency=1000:sample_rate=44100:duration=6,volume='if(lt(mod(t,0.5),0.05),0.3,0)':eval=frame"
    # chorus: loud clicks every 0.5s (drum-like) mixed with a sustained
    # 440Hz tone pulsed every 0.25s (guitar-note-like), both full volume
    chorus_drums = "sine=frequency=1000:sample_rate=44100:duration=7,volume='if(lt(mod(t,0.5),0.05),1,0)':eval=frame"
    chorus_notes = "sine=frequency=440:sample_rate=44100:duration=7,volume='if(lt(mod(t,0.25),0.08),0.9,0)':eval=frame"
    outro = "sine=frequency=1000:sample_rate=44100:duration=5,volume='if(lt(mod(t,0.5),0.05),0.3,0)':eval=frame"
    _ffmpeg(
        "-f", "lavfi", "-i", verse,
        "-f", "lavfi", "-i", chorus_drums,
        "-f", "lavfi", "-i", chorus_notes,
        "-f", "lavfi", "-i", outro,
        "-filter_complex",
        "[1:a][2:a]amix=inputs=2:duration=first[chorus];"
        "[0:a][chorus][3:a]concat=n=3:v=0:a=1[out]",
        "-map", "[out]", path,
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


@pytest.fixture(scope="session")
def similar_shots_video(media_dir):
    """A ~4.5s video with hard cuts between NEARLY IDENTICAL shots (same
    subject, only a slight color-tint change each cut) — mimics the reference
    edit's mask-with-different-mouth-colors sequence that a motion-robust
    detector (AdaptiveDetector) is prone to miss entirely, since its whole
    design point is ignoring small frame-to-frame differences."""
    path = str(media_dir / "similar_shots.mp4")
    n = 9
    inputs = []
    graph = []
    for i in range(n):
        tint = 0x10 * i  # subtle, monotonically shifting tint per cut
        color = f"0x{tint:02x}2020"
        inputs += ["-f", "lavfi", "-i", f"color=c={color}:size=320x568:rate=30:duration=0.5"]
        graph.append(f"[{i}:v]format=yuv420p[v{i}];")
    labels = "".join(f"[v{i}]" for i in range(n))
    graph.append(f"{labels}concat=n={n}:v=1:a=0[outv]")
    _ffmpeg(
        *inputs,
        "-filter_complex", "".join(graph),
        "-map", "[outv]",
        "-c:v", "libx264",
        path,
    )
    return path
