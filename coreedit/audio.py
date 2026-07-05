"""Song rhythm analysis via librosa: beats, onsets, tempo, energy."""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

from .models import BeatGrid

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


def _load_audio(path: str, start: float, duration: float | None):
    """Load mono audio via librosa; video containers go through an ffmpeg
    wav extraction first (soundfile can't read them and librosa's audioread
    fallback is deprecated)."""
    import librosa

    if os.path.splitext(path)[1].lower() in AUDIO_EXTS:
        return librosa.load(path, sr=None, mono=True, offset=start, duration=duration)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-ss", str(start), "-i", path]
        if duration is not None:
            cmd += ["-t", str(duration)]
        cmd += ["-vn", "-ac", "1", tmp]
        subprocess.run(cmd, check=True)
        return librosa.load(tmp, sr=None, mono=True)
    finally:
        os.unlink(tmp)


def analyze_song(
    path: str,
    start: float = 0.0,
    duration: float | None = None,
) -> BeatGrid:
    """Analyze the rhythm of an audio (or video) file.

    Times in the returned BeatGrid are relative to `start`, i.e. they map
    directly onto the output edit's timeline.
    """
    import librosa

    y, sr = _load_audio(path, start, duration)
    if y.size == 0:
        raise ValueError(f"no audio decoded from {path} at offset {start}s")
    total = float(len(y)) / sr

    tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units="time")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, units="time", backtrack=False
    )

    rms = librosa.feature.rms(y=y)[0]
    peak = float(rms.max()) if rms.size else 0.0
    energy = (rms / peak).tolist() if peak > 0 else rms.tolist()
    hop = 512  # librosa default for rms/onset frames
    energy_times = (np.arange(len(rms)) * hop / sr).tolist()

    bpm = float(np.atleast_1d(tempo)[0])
    return BeatGrid(
        bpm=bpm,
        beat_times=[float(t) for t in beat_times],
        onset_times=[float(t) for t in onset_times],
        energy_times=energy_times,
        energy=energy,
        duration=total,
    )


def snap_to_grid(value: float, candidates: list[float], tolerance: float) -> float:
    """Snap `value` to the nearest candidate time if within tolerance.

    Candidates must be sorted ascending. Returns `value` unchanged when no
    candidate is close enough (or the list is empty).
    """
    if not candidates:
        return value
    import bisect

    i = bisect.bisect_left(candidates, value)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(candidates):
            if best is None or abs(candidates[j] - value) < abs(best - value):
                best = candidates[j]
    if best is not None and abs(best - value) <= tolerance:
        return best
    return value
