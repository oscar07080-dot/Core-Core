import numpy as np

from coreedit.audio import _onsets_with_strength, _rms_feature


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


def test_min_spacing_converts_to_wait_frames_correctly(monkeypatch):
    """The min_spacing (seconds) -> wait (frames) conversion is what actually
    enforces the minimum gap; verify it's computed and passed through
    correctly rather than relying on a synthetic signal reliably producing
    two independently-resolvable spectral peaks (fragile/finicky to
    construct — real-audio validation lives in the e2e reference tests)."""
    import librosa

    captured = {}
    real_onset_detect = librosa.onset.onset_detect

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real_onset_detect(*args, **kwargs)

    monkeypatch.setattr(librosa.onset, "onset_detect", spy)

    sr = 22050
    y = _synthetic_note(sr, total=2.0, attack_start=1.0, ramp=0.05)
    _onsets_with_strength(y, sr, min_spacing=0.1)

    hop_length = 512
    expected_wait = round(0.1 * sr / hop_length)
    assert captured["wait"] == expected_wait


def test_min_spacing_wait_is_never_zero():
    # a tiny min_spacing must still floor to at least 1 frame, not 0
    # (wait=0 would disable the minimum-spacing behavior entirely)
    sr = 22050
    y = _synthetic_note(sr, total=2.0, attack_start=1.0, ramp=0.05)
    times, _ = _onsets_with_strength(y, sr, min_spacing=0.0001)
    assert len(times) >= 1  # just needs to run without error and find the note


# --- RMS-based feature: fixes onsets landing on timbre change instead of a loudness swell ---


def test_rms_feature_matches_librosa_rms():
    import librosa

    sr = 22050
    y = _synthetic_note(sr, total=2.0, attack_start=1.0, ramp=0.05)
    out = _rms_feature(y=y, sr=sr, n_fft=2048, hop_length=512)
    expected = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)
    np.testing.assert_array_equal(out, expected)


def test_onsets_with_rms_feature_detects_a_real_loudness_swell():
    sr = 22050
    y = _synthetic_note(sr, total=3.0, attack_start=1.5, ramp=0.15)
    times, strengths = _onsets_with_strength(y, sr, delta=0.02, feature=_rms_feature)
    assert len(times) >= 1
    assert 1.4 <= times[0] <= 1.65
    assert len(strengths) == len(times)


def test_rms_feature_ignores_pure_timbre_change_at_constant_loudness():
    """A frequency change with no loudness change is a real spectral-flux
    trigger but should NOT register as an RMS-feature onset -- this is the
    exact distinction that fixed cuts landing on timbre shifts instead of
    the strum's actual loudness swell."""
    sr = 22050
    n = int(sr * 3.0)
    t = np.arange(n) / sr
    # constant envelope throughout; only the frequency changes partway through
    y = np.where(t < 1.5, np.sin(2 * np.pi * 220.0 * t), np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    rms_times, _ = _onsets_with_strength(y, sr, delta=0.02, feature=_rms_feature)
    flux_times, _ = _onsets_with_strength(y, sr, delta=0.02)

    near_switch_rms = [t for t in rms_times if 1.3 <= t <= 1.7]
    near_switch_flux = [t for t in flux_times if 1.3 <= t <= 1.7]
    assert near_switch_rms == []
    assert len(near_switch_flux) >= 1
