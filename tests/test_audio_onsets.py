import numpy as np

from coreedit.audio import _onsets_with_strength


def _synthetic_note(sr: int, total: float, attack_start: float, ramp: float, freq: float = 440.0):
    """A tone that ramps up from silence to full volume over `ramp` seconds
    starting at `attack_start` — mimics a plucked/strummed note's gradual
    attack, unlike a percussive click's near-instant one."""
    n = int(sr * total)
    t = np.arange(n) / sr
    envelope = np.clip((t - attack_start) / ramp, 0, 1)
    envelope[t < attack_start] = 0.0
    return (envelope * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_backtracked_onset_lands_at_or_before_the_attack_ramp():
    sr = 22050
    attack_start, ramp = 1.5, 0.15
    y = _synthetic_note(sr, total=3.0, attack_start=attack_start, ramp=ramp)

    times, strengths = _onsets_with_strength(y, sr)

    assert len(times) >= 1
    onset_time = times[0]
    # a peak-only detector would report the end of the ramp (envelope at 1.0);
    # backtracking must pull that back towards where the ramp actually starts
    assert attack_start - 0.1 <= onset_time <= attack_start + ramp
    assert len(strengths) == len(times)
    assert strengths[0] > 0


def test_silence_produces_no_onsets():
    sr = 22050
    y = np.zeros(int(sr * 2.0), dtype=np.float32)
    times, strengths = _onsets_with_strength(y, sr)
    assert times == []
    assert strengths == []


def test_lower_delta_catches_quieter_onsets():
    sr = 22050
    # two notes: one full-strength, one much quieter
    y1 = _synthetic_note(sr, total=4.0, attack_start=1.0, ramp=0.05)
    y2 = 0.15 * _synthetic_note(sr, total=4.0, attack_start=2.5, ramp=0.05)
    y = y1 + y2

    strict_times, _ = _onsets_with_strength(y, sr, delta=0.07)
    lenient_times, _ = _onsets_with_strength(y, sr, delta=0.01)
    assert len(lenient_times) >= len(strict_times)


def test_strength_reflects_peak_not_backtracked_quiet_point():
    sr = 22050
    y = _synthetic_note(sr, total=3.0, attack_start=1.0, ramp=0.2)
    times, strengths = _onsets_with_strength(y, sr)
    assert len(strengths) >= 1
    # strength must be sampled near the note's full volume, not near-zero
    # (which is what the backtracked, pre-attack point would give)
    assert strengths[0] > 0.01
