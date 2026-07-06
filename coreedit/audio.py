"""Song rhythm analysis via librosa: beats, onsets, tempo, energy."""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

from .models import BeatGrid, SectionOverride

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
    note_sensitivity: float = 0.05,
    note_min_spacing: float = 0.1,
) -> BeatGrid:
    """Analyze the rhythm of an audio (or video) file.

    Times in the returned BeatGrid are relative to `start`, i.e. they map
    directly onto the output edit's timeline.

    `note_sensitivity` is the peak-picking threshold for melodic (guitar-like)
    onsets — lower catches more/quieter notes, higher misses more of them.
    Librosa's own default (0.07) is tuned for full-mix/percussive-style sharp
    transients; a separated harmonic stream's softer attacks warrant a lower
    default here.

    `note_min_spacing` is the minimum time between two melodic onsets.
    Librosa's own default minimum spacing is ~30ms, which is far shorter than
    any genuinely distinct guitar note at normal playing speed — a single
    sustained or vibrato note's natural energy flutter gets picked up as 2-3
    separate "onsets" a few frames apart. Measured against a real reference
    edit's actual cuts, that produced ~2.5x as many detected onsets as real
    notes. 100ms (10 notes/sec) is fast even for a real guitar line while
    still rejecting that within-note flutter.
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

    # split into melodic (guitar/vocal-like sustained tones) vs percussive
    # (drum hits) components so cuts can follow one or the other on request
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    # a plucked/strummed note rings up gradually, so its onset-strength peak
    # lags well behind the actual attack; a drum hit is already sharp enough
    # that this barely matters, so only the harmonic stream needs the lower
    # sensitivity threshold too.
    harmonic_times, harmonic_strength = _onsets_with_strength(
        y_harmonic, sr, delta=note_sensitivity, min_spacing=note_min_spacing
    )
    percussive_times, percussive_strength = _onsets_with_strength(y_percussive, sr)

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
        harmonic_onset_times=harmonic_times,
        percussive_onset_times=percussive_times,
        harmonic_onset_strength=harmonic_strength,
        percussive_onset_strength=percussive_strength,
    )


def _onsets_with_strength(
    y: np.ndarray, sr: int, delta: float = 0.07, min_spacing: float = 0.03
) -> tuple[list[float], list[float]]:
    """Onset times paired with how pronounced each one is.

    Peak-picking finds each onset's strongest point, which for a note that
    rings up gradually (a strum, not a sharp transient) sits noticeably after
    the actual attack. `onset_backtrack` rolls each detected peak back to the
    preceding local minimum, which tracks the true attack much more closely.
    Strength is still sampled at the original peak (not the backtracked,
    necessarily-quiet point) so accented vs. barely-there onsets stay
    distinguishable for downstream thinning.

    `min_spacing` (seconds) is the minimum gap enforced between two onsets —
    librosa's own default is ~30ms (`wait` in frames), which readily lets a
    single note's natural energy flutter register as several onsets.
    """
    import librosa

    hop_length = 512
    wait = max(1, round(min_spacing * sr / hop_length))
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    peak_frames = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, units="frames", backtrack=False,
        delta=delta, wait=wait, hop_length=hop_length,
    )
    if len(peak_frames) == 0:
        return [], []
    attack_frames = librosa.onset.onset_backtrack(peak_frames, env)
    times = librosa.frames_to_time(attack_frames, sr=sr, hop_length=hop_length)
    strengths = env[peak_frames]
    return [float(t) for t in times], [float(s) for s in strengths]


def _smooth(values: list[float], window_samples: int) -> list[float]:
    """Centered moving average, used to turn the frame-level energy curve
    into a section-level contour before looking for sustained plateaus."""
    n = len(values)
    if n == 0 or window_samples <= 1:
        return list(values)
    half = window_samples // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def detect_chorus_and_buildup(
    grid: BeatGrid,
    min_chorus_length: float = 6.0,
    buildup_length: float = 4.0,
    threshold_frac: float = 0.55,
) -> tuple[SectionOverride | None, SectionOverride | None]:
    """Best-effort heuristic: the chorus is treated as the longest sustained
    high-energy stretch of the song, and the build-up is the `buildup_length`
    seconds immediately before it. Returns (None, None) if nothing in the
    analyzed audio qualifies (e.g. no clear plateau, or it's too short)."""
    if not grid.energy_times:
        return None, None

    dt = (
        (grid.energy_times[-1] - grid.energy_times[0]) / (len(grid.energy_times) - 1)
        if len(grid.energy_times) > 1 else 0.1
    )
    window_samples = max(1, round(1.5 / dt)) if dt > 0 else 1
    smoothed = _smooth(grid.energy, window_samples)
    peak = max(smoothed) if smoothed else 0.0
    if peak <= 0:
        return None, None
    threshold = peak * threshold_frac

    runs: list[tuple[int, int]] = []
    run_start = None
    for i, v in enumerate(smoothed):
        if v >= threshold:
            run_start = i if run_start is None else run_start
        elif run_start is not None:
            runs.append((run_start, i - 1))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(smoothed) - 1))

    qualifying = [
        (s, e) for s, e in runs
        if grid.energy_times[e] - grid.energy_times[s] >= min_chorus_length
    ]
    if not qualifying:
        return None, None

    best_s, best_e = max(
        qualifying,
        key=lambda se: sum(smoothed[se[0]:se[1] + 1]) / (se[1] - se[0] + 1),
    )
    chorus_start, chorus_end = grid.energy_times[best_s], grid.energy_times[best_e]
    chorus = SectionOverride(start=chorus_start, end=chorus_end, kind="harmonic")

    buildup_start = max(0.0, chorus_start - buildup_length)
    buildup = (
        SectionOverride(start=buildup_start, end=chorus_start, kind="percussive")
        if chorus_start - buildup_start >= 1.0
        else None
    )
    return chorus, buildup


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
