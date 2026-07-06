import math

import pytest

from coreedit.models import BeatGrid, Clip


@pytest.fixture
def steady_grid() -> BeatGrid:
    """120 BPM grid over 20s: beats every 0.5s, onsets every 0.25s,
    energy ramps 0 -> 1 over the duration."""
    beats = [round(i * 0.5, 4) for i in range(40)]
    onsets = [round(i * 0.25, 4) for i in range(80)]
    n = 200
    times = [i * 0.1 for i in range(n)]
    energy = [i / (n - 1) for i in range(n)]
    # distinct from beats/onsets so override tests can tell them apart
    harmonic = [round(i * 0.3 + 0.02, 4) for i in range(67)]
    percussive = [round(i * 0.15 + 0.01, 4) for i in range(133)]
    return BeatGrid(
        bpm=120.0,
        beat_times=beats,
        onset_times=onsets,
        energy_times=times,
        energy=energy,
        duration=20.0,
        harmonic_onset_times=harmonic,
        percussive_onset_times=percussive,
    )


@pytest.fixture
def clip_pool() -> list[Clip]:
    return [
        Clip(path="/fake/a.mp4", duration=10.0),
        Clip(path="/fake/b.mp4", duration=8.0),
        Clip(path="/fake/c.mp4", duration=15.0),
        Clip(path="/fake/d.png", duration=math.inf, is_image=True),
    ]
