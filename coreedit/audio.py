"""Song rhythm analysis via librosa: beats, onsets, tempo, energy."""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

from .models import BeatGrid, SectionOverride

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus"}

_MADMOM = None  # cached import result: module when available, False when not


def _madmom():
    """Import madmom if installed, applying compatibility shims first.

    madmom 0.16.1 (the last PyPI release) predates Python 3.10 and numpy 2:
    it imports MutableSequence from `collections` (moved to collections.abc)
    and uses the removed np.int/np.float/np.bool aliases -- including inside
    its *compiled* Cython modules, where the source can't be patched.
    Injecting the old names before import fixes both on a pristine install.
    Returns the module, or False when madmom isn't installed.
    """
    global _MADMOM
    if _MADMOM is not None:
        return _MADMOM
    try:
        import collections
        import collections.abc

        for name in ("MutableSequence", "MutableMapping", "Iterable", "Callable"):
            if not hasattr(collections, name):
                setattr(collections, name, getattr(collections.abc, name))
        for alias, typ in (("int", int), ("float", float), ("bool", bool)):
            if not hasattr(np, alias):
                setattr(np, alias, typ)
        import madmom  # noqa: F401

        _MADMOM = madmom
    except ImportError:
        _MADMOM = False
    return _MADMOM


def _madmom_onsets(y: np.ndarray, sr: int, min_spacing: float) -> tuple[list[float], list[float]]:
    """Neural (CNN) onset detection: times plus per-onset activation strength.

    The CNN reports each onset at its annotated attack time directly (10ms
    resolution), so no backtracking is needed, and its activation value is a
    0..1 confidence that doubles as an accent-strength measure.
    """
    from madmom.audio.signal import Signal
    from madmom.features.onsets import CNNOnsetProcessor, OnsetPeakPickingProcessor

    sig = Signal(y.astype(np.float32), sample_rate=sr)
    act = CNNOnsetProcessor()(sig)
    picker = OnsetPeakPickingProcessor(
        fps=100, threshold=0.3, smooth=0.05, pre_max=0.05, post_max=0.05,
        combine=max(min_spacing, 0.01),
    )
    times = picker(act)
    strengths = [float(act[min(int(round(t * 100)), len(act) - 1)]) for t in times]
    return [float(t) for t in times], strengths


def _normalize_strengths(strengths: list[float]) -> list[float]:
    """Scale a strength list to 0..1 by its own max, so onset strengths from
    different detectors (madmom's 0..1 CNN activation vs librosa's arbitrary
    RMS-derivative units) become comparable before merging."""
    peak = max(strengths) if strengths else 0.0
    return [s / peak for s in strengths] if peak > 0 else list(strengths)


def _merge_onsets(
    times_a: list[float], strengths_a: list[float],
    times_b: list[float], strengths_b: list[float],
    tolerance: float = 0.05,
) -> tuple[list[float], list[float]]:
    """Union two onset detections, normalizing each to its own 0..1 scale
    first and treating detections within `tolerance` seconds as the same
    onset (keeping the higher of the two normalized strengths).

    Different onset detectors miss different real notes -- checked directly
    against a real song's waveform, madmom's CNN onset detector had a
    complete blind spot exactly where the two loudest, most obvious swells
    in one section were (missed entirely, no onset at all), while librosa's
    simpler RMS-derivative approach caught both; librosa in turn misses
    subtler attacks madmom catches elsewhere. Combining catches what either
    one alone misses.
    """
    a = _normalize_strengths(strengths_a)
    b = _normalize_strengths(strengths_b)
    combined = sorted(zip(times_a, a)) + sorted(zip(times_b, b))
    combined.sort(key=lambda pair: pair[0])

    merged_times: list[float] = []
    merged_strengths: list[float] = []
    for t, s in combined:
        if merged_times and t - merged_times[-1] <= tolerance:
            if s > merged_strengths[-1]:
                merged_times[-1] = t
                merged_strengths[-1] = s
        else:
            merged_times.append(t)
            merged_strengths.append(s)
    return merged_times, merged_strengths


def _madmom_beats(y: np.ndarray, sr: int) -> tuple[list[float], float]:
    """Neural beat tracking (RNN + DBN): beat times plus derived BPM."""
    from madmom.audio.signal import Signal
    from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor

    sig = Signal(y.astype(np.float32), sample_rate=sr)
    beats = DBNBeatTrackingProcessor(fps=100)(RNNBeatProcessor()(sig))
    beat_times = [float(b) for b in beats]
    if len(beat_times) >= 2:
        bpm = 60.0 / float(np.median(np.diff(beat_times)))
    else:
        bpm = 0.0
    return beat_times, bpm


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
    note_sensitivity: float = 0.02,
    note_min_spacing: float = 0.1,
) -> BeatGrid:
    """Analyze the rhythm of an audio (or video) file.

    Times in the returned BeatGrid are relative to `start`, i.e. they map
    directly onto the output edit's timeline.

    When madmom is installed it's used for beat tracking, percussive onsets,
    and as one of two detectors merged for melodic (harmonic) onsets — its
    CNN detector is generally more accurate than librosa's heuristics, but
    checked directly against a real song's waveform it had a complete blind
    spot for two of the loudest swells in one section, which librosa's
    RMS-derivative approach caught; see _merge_onsets. Without madmom,
    librosa's detector is used alone.

    `note_sensitivity` is the peak-picking threshold for the librosa melodic
    (guitar-like) onset detector — lower catches more/quieter notes, higher
    misses more of them. Tuned for its RMS-derivative novelty curve (see
    `_onsets_with_strength`).

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

    # split into melodic (guitar/vocal-like sustained tones) vs percussive
    # (drum hits) components so cuts can follow one or the other on request
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Guitar/melodic "onsets" are cut points meant to land on a strum's
    # loudness swell. This RMS-derivative novelty curve (attack = where
    # loudness is actually rising) tracks that far better than spectral flux,
    # which responds to timbre change, not loudness -- checked visually
    # against a real reference edit's audio and roughly half of spectral
    # flux's detections landed in flat/declining regions with no audible
    # swell nearby.
    librosa_harmonic_times, librosa_harmonic_strength = _onsets_with_strength(
        y_harmonic, sr, delta=note_sensitivity, min_spacing=note_min_spacing,
        feature=_rms_feature,
    )

    if _madmom():
        # madmom's neural CNN onset detector and RNN+DBN beat tracker are
        # generally more accurate than the librosa heuristics -- but checked
        # directly against a real song's waveform, its CNN missed two of the
        # loudest, most obvious swells in one section entirely (no onset at
        # all), which librosa's RMS-derivative approach caught. Neither
        # detector alone is reliable; merge both so each catches what the
        # other misses (see _merge_onsets). Percussive (drum) hits are sharp,
        # unambiguous transients where this blind-spot risk hasn't shown up,
        # so that stream stays madmom-only. Optional because installing
        # madmom is nontrivial (see README).
        beat_times, bpm = _madmom_beats(y, sr)
        onset_times, _ = _madmom_onsets(y, sr, min_spacing=0.03)
        madmom_harmonic_times, madmom_harmonic_strength = _madmom_onsets(
            y_harmonic, sr, min_spacing=note_min_spacing
        )
        harmonic_times, harmonic_strength = _merge_onsets(
            madmom_harmonic_times, madmom_harmonic_strength,
            librosa_harmonic_times, librosa_harmonic_strength,
        )
        percussive_times, percussive_strength = _madmom_onsets(
            y_percussive, sr, min_spacing=0.03
        )
    else:
        tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units="time")
        beat_times = [float(t) for t in beat_times]
        bpm = float(np.atleast_1d(tempo)[0])
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_times = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, units="time", backtrack=False
        )
        onset_times = [float(t) for t in onset_times]
        harmonic_times, harmonic_strength = librosa_harmonic_times, librosa_harmonic_strength
        percussive_times, percussive_strength = _onsets_with_strength(y_percussive, sr)

    rms = librosa.feature.rms(y=y)[0]
    peak = float(rms.max()) if rms.size else 0.0
    energy = (rms / peak).tolist() if peak > 0 else rms.tolist()
    hop = 512  # librosa default for rms/onset frames
    energy_times = (np.arange(len(rms)) * hop / sr).tolist()

    return BeatGrid(
        bpm=bpm,
        beat_times=beat_times,
        onset_times=onset_times,
        energy_times=energy_times,
        energy=energy,
        duration=total,
        harmonic_onset_times=harmonic_times,
        percussive_onset_times=percussive_times,
        harmonic_onset_strength=harmonic_strength,
        percussive_onset_strength=percussive_strength,
    )


def _rms_feature(y=None, sr=None, n_fft=2048, hop_length=512, **kwargs):
    """Adapts librosa.feature.rms to the (y, sr, n_fft, hop_length) signature
    librosa.onset.onset_strength expects of a `feature` callable -- rms()
    itself takes frame_length, not n_fft/sr, and doesn't accept extra kwargs.
    """
    import librosa

    return librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)


def _onsets_with_strength(
    y: np.ndarray, sr: int, delta: float = 0.07, min_spacing: float = 0.03,
    feature=None,
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

    `feature` overrides the novelty-curve basis passed to
    librosa.onset.onset_strength (defaults to its own spectral-flux feature).
    """
    import librosa

    hop_length = 512
    wait = max(1, round(min_spacing * sr / hop_length))
    kwargs = {"feature": feature} if feature is not None else {}
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, **kwargs)
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
