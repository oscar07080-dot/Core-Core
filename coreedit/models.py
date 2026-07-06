"""Data models shared across the pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Clip:
    """A source video clip or still image available for the edit."""

    path: str
    duration: float  # seconds; math.inf for still images
    is_image: bool = False

    def usable_range(self, margin: float) -> float:
        """Seconds of footage available after trimming margin off both ends."""
        if self.is_image:
            return math.inf
        return self.duration - 2 * margin


@dataclass
class Segment:
    """One cut in the final timeline: a slice of a source clip."""

    clip: Clip
    in_point: float   # seconds into the source clip (0.0 for images)
    duration: float   # seconds on the output timeline

    @property
    def out_point(self) -> float:
        return self.in_point + self.duration


@dataclass
class BeatGrid:
    """Rhythm analysis of a song (or of a reference edit's audio track)."""

    bpm: float
    beat_times: list[float]      # seconds, coarse beat grid
    onset_times: list[float]     # seconds, dense transient grid (full mix)
    energy_times: list[float]    # seconds, sample times of the energy curve
    energy: list[float]          # normalized 0..1 RMS energy at energy_times
    duration: float              # seconds of analyzed audio
    harmonic_onset_times: list[float] = field(default_factory=list)   # melodic note onsets (guitar, etc.)
    percussive_onset_times: list[float] = field(default_factory=list)  # drum-hit onsets
    harmonic_onset_strength: list[float] = field(default_factory=list)   # how pronounced each harmonic_onset_times entry is
    percussive_onset_strength: list[float] = field(default_factory=list)  # how pronounced each percussive_onset_times entry is

    @property
    def beat_period(self) -> float:
        return 60.0 / self.bpm if self.bpm > 0 else 0.5

    def energy_at(self, t: float) -> float:
        """Energy value nearest to time t (0.0 if no energy data)."""
        if not self.energy_times:
            return 0.0
        # energy_times is sorted and evenly spaced; nearest index by ratio
        span = self.energy_times[-1] - self.energy_times[0]
        if span <= 0:
            return self.energy[0]
        frac = (t - self.energy_times[0]) / span
        idx = round(frac * (len(self.energy_times) - 1))
        idx = max(0, min(len(self.energy) - 1, idx))
        return self.energy[idx]


@dataclass
class RhythmTemplate:
    """Cut pacing extracted from a reference edit, expressed in beat units."""

    cut_lengths_in_beats: list[float]
    source_bpm: float

    def durations_for_bpm(self, bpm: float) -> list[float]:
        """Rescale the template's cut lengths to a new tempo, in seconds."""
        period = 60.0 / bpm if bpm > 0 else 0.5
        return [n * period for n in self.cut_lengths_in_beats]


@dataclass
class SectionOverride:
    """A time range where cuts should follow a specific onset stream instead
    of the normal pacing: "harmonic" cuts on every melodic note (e.g. guitar),
    "percussive" cuts on every drum hit."""

    start: float
    end: float
    kind: str  # "harmonic" or "percussive"


@dataclass
class EditSpec:
    """Resolved settings for one edit run."""

    song: str
    output: str
    duration: float = 20.0
    song_start: float = 0.0
    width: int = 1080
    height: int = 1920
    fps: int = 30
    intensity: str = "medium"
    no_repeat_window: int = 3
    source_margin: float = 0.5
    seed: int | None = None
    clips: list[Clip] = field(default_factory=list)
