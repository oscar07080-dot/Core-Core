# Core-Core

CLI tool that generates TikTok-ready, beat-synced montage edits ("core edits"):
hard cuts on the song's beat, near-strobe cutting in high-energy sections,
vertical 1080x1920 output with the song as the only audio.

## Setup

Requires Python 3.10+ and the `ffmpeg`/`ffprobe` binaries on your PATH.

```bash
pip install -r requirements.txt
```

### Optional: madmom (recommended for noticeably better cut timing)

When [madmom](https://github.com/CPJKU/madmom) is importable, its neural
beat tracker and CNN onset detector are used for beat tracking, drum-hit
onsets, and as one of two detectors merged for melodic (guitar-like)
onsets. Neither detector alone is reliable: audited directly against a real
song's waveform, madmom's CNN had a complete blind spot for the three
loudest, most obvious swells in one section (no onset at all), which
librosa's simpler RMS-derivative detector caught; librosa in turn misses
subtler attacks madmom catches elsewhere. The two are merged (normalized to
a comparable scale, deduplicated where they agree) so melodic cuts catch
what either one alone misses. The tool applies the needed Python 3.10+/
numpy 2 compatibility shims automatically at import, so the old 0.16.1
release works unpatched. Installing it is the only fiddly part:

```bash
pip install cython
# on Debian/Ubuntu, the distro-patched setuptools breaks the build; replace it:
pip install --ignore-installed setuptools wheel
pip install madmom
```

If the install fails, everything still works on the librosa fallback —
`edit.py` prints which backend it's using.

## Usage

Sources come from **either** a directory of local files **or** a list of URLs
you provide (downloaded with yt-dlp) — pick one per run.

```bash
# local clips (videos and/or still images)
python edit.py --song song.mp3 --clips-dir ./clips --subject "my subject"

# clips downloaded from URLs you supply
python edit.py --song song.mp3 --urls urls.txt --subject "my subject"

# copy the cut pacing of an existing edit
python edit.py --song song.mp3 --clips-dir ./clips --reference some_edit.mp4

# preview the cut list without rendering
python edit.py --song song.mp3 --clips-dir ./clips --dry-run

# cut on every guitar note through the chorus, every drum hit right before it
python edit.py --song song.mp3 --clips-dir ./clips --auto-chorus
python edit.py --song song.mp3 --clips-dir ./clips --chorus 45:60 --drum-buildup 40:45

# export as numbered clips instead of one rendered mp4 (see below)
python edit.py --song song.mp3 --clips-dir ./clips --export-clips ./for_capcut

# or build a native CapCut project directly in CapCut's drafts folder
python edit.py --song song.mp3 --clips-dir ./clips --export-capcut my_edit
```

### Editing the result yourself (e.g. in CapCut)

Two options, most-automatic first:

**`--export-capcut [NAME]`** builds a native CapCut desktop draft
(project) straight into CapCut's drafts folder: the full timeline
pre-built — one pre-trimmed clip per cut, the song on the audio track,
all media stored inside the draft folder. Open CapCut (restart it if it
was already running) and the project is on the home screen; select each
clip, hit **Replace**, and pick your own footage — Replace keeps the
slot's duration, so the cut timing is untouched. Requires the optional
`pyJianYingDraft` package (`pip install pyJianYingDraft`) and CapCut
**desktop** (drafts on mobile live in the app's private sandbox — no
import path). The drafts folder is auto-detected in the standard
Windows/macOS locations; if yours is elsewhere (CapCut: Settings →
Drafts shows the path), pass `--capcut-drafts-dir DIR`. **Caveat**: the
draft format is undocumented and reverse-engineered, and newer CapCut
versions have been reported to encrypt drafts — if the project doesn't
appear or won't open, fall back to `--export-clips`. The draft must be
generated directly into the real drafts folder on the machine running
CapCut (media is referenced by absolute path), so run the tool on that
machine rather than copying a draft folder over.

**`--export-clips DIR`** (works with every editor/version) writes each
segment as its own numbered file (`001.mp4`, `002.mp4`, ...) plus the
trimmed song audio (`song.m4a`) into `DIR`, instead of rendering one
mp4. Drag the numbered clips into a timeline in order, add `song.m4a`
as the audio track, then use your editor's **"replace clip"** feature on
each one to swap in your own footage — it keeps the same trim/duration,
so the cut timing survives untouched. Works in CapCut (any version) and
any other NLE that can replace a clip in place.

### URL list format (`urls.txt`)

One source per line: `URL|START|END` (start/end in seconds, both optional —
when given, only that section is downloaded). `#` starts a comment.

```
https://www.youtube.com/watch?v=XXXXXXXXXXX
https://www.youtube.com/watch?v=YYYYYYYYYYY|90|120
# a comment
```

Downloads are cached in `--cache-dir` (default `./cache`); re-runs reuse them.

If you're behind a proxy, set `HTTPS_PROXY` (or `ALL_PROXY`) as usual — it's
picked up automatically and passed through explicitly to the trimmed-download
path, which shells out to ffmpeg and would otherwise silently bypass the
proxy (ffmpeg doesn't read `HTTPS_PROXY` on its own the way most tools do).

### Chorus / drum-buildup section overrides

By default cuts follow the beat/onset pacing (or a `--reference`'s rhythm)
uniformly across the whole edit. You can override that in specific time
ranges to cut on every melodic note ("harmonic" onsets — guitar, etc.) or
every drum hit ("percussive" onsets), using librosa's harmonic/percussive
source separation:

- `--chorus START:END` (repeatable) — cut on every melodic note in that range
- `--drum-buildup START:END` (repeatable) — cut on every drum hit in that range
- `--auto-chorus` — instead of specifying ranges, auto-detect the chorus as
  the longest sustained high-energy stretch of the song, and treat the 4s
  right before it as the drum build-up. Best-effort heuristic — if it picks
  the wrong section (or none), fall back to explicit `--chorus`/`--drum-buildup`
  timestamps, which always take precedence over `--auto-chorus`.

Cut selection inside these ranges is **deterministic**: a note/hit gets a
cut if and only if its strength is at least `1 - variation` of the loudest
onset within 2 seconds of it, so the same musical phrase always produces
the same cut pattern and every accent reliably cuts. (An earlier version
used a strength-weighted coin flip per note; that meant a repeated riff cut
differently on each repetition and accents sometimes didn't cut at all —
individually beat-aligned but rhythmically arbitrary.) The comparison is
local, not against the whole range's single loudest peak, so a quiet
passage's own accents still cut. A max-gap rule additionally guarantees no
stretch inside a range goes more than ~2 beats without a cut (the strongest
skipped note in an oversized gap is promoted). The first note of a rising
phrase (a step up into a louder follow-on note) is always kept too, even if
it fails the loudness threshold on its own — a phrase climbing to a peak is
typically voiced quietest at its launch, so a pure loudness rule otherwise
drops exactly the note marking where the rise begins. `--seed` affects only
which clips fill the segments, never cut timing.

Tune with `--section-variation` (0 = only exact local peaks cut; 1 = every
single note cuts; default 0.35) and, separately, `--drum-variation` for the
`--drum-buildup` ranges (1 = cut on every drum hit — the fastest; defaults
to `--section-variation`'s value).

Melodic notes are detected from the harmonic stream's RMS (loudness) rise,
not spectral flux — checked visually against a real reference edit (plotting
the waveform/amplitude envelope with detected onsets overlaid, zoomed to
sub-second windows): spectral flux responds to *timbre* change and roughly
half its detections landed in flat/declining regions with no audible swell
nearby, while the loudness-rise approach tracked the real strums far more
closely. Timing is then backtracked from each note's loudest point to its
actual attack (a strummed/plucked note rings up gradually, so the two can be
tens of milliseconds apart). If it's still missing real notes, lower
`--note-sensitivity` (default 0.02); if it's over-triggering on sustain/
vibrato, raise it.

Detected notes are also required to be at least `--note-min-spacing`
(default 0.1s) apart — librosa's own default minimum gap is ~30ms, which is
short enough that a single sustained/vibrato note's natural energy flutter
gets picked up as 2-3 separate onsets. Checked against a real reference
edit's actual cuts, that produced ~2.5x as many detected notes as the
editor's real cuts; 0.1s brought it down to ~1.6x. Lower it if your song's
guitar part is genuinely faster than 10 notes/sec, raise it if it's still
over-triggering.

### Key flags

| flag | default | meaning |
|---|---|---|
| `--duration` | 20 | edit length in seconds |
| `--song-start` | 0 | offset into the song (pick the good part) |
| `--reference` | — | edit video whose cut rhythm is copied |
| `--scene-threshold` | 12.0 | cut-detection sensitivity for `--reference` (lower = catches more subtle cuts, risks false positives) |
| `--intensity` | medium | cut density (`low`/`medium`/`high`) when no reference |
| `--chorus` | — | `START:END` range to cut on every melodic note (repeatable) |
| `--note-sensitivity` | 0.02 | melodic-note detection sensitivity (lower = catches more/quieter notes) |
| `--note-min-spacing` | 0.1 | minimum seconds between two detected melodic notes |
| `--drum-buildup` | — | `START:END` range to cut on every drum hit (repeatable) |
| `--auto-chorus` | off | auto-detect the chorus/build-up instead of specifying ranges |
| `--section-variation` | 0.35 | pacing variation within `--chorus` ranges (0=accents only, 1=uniform) |
| `--drum-variation` | (same as above) | pacing within `--drum-buildup` ranges (1 = cut every drum hit, fastest) |
| `--seed` | — | reproducible clip selection |
| `--source-margin` | 0.5 | seconds skipped at each clip's start/end |
| `--no-repeat-window` | 3 | recent clips excluded from reuse |
| `--export-clips` | — | export numbered per-cut clips + song instead of one mp4 |
| `--export-capcut` | — | build a native CapCut draft in CapCut's drafts folder |
| `--capcut-drafts-dir` | auto-detect | CapCut drafts folder location for `--export-capcut` |
| `--output` | `{subject}_{timestamp}.mp4` | output path |

## How it works

1. **Acquire** — yt-dlp downloads (section-only when trimmed) or local globbing; still images (`jpg/png/webp`) are valid sources and become brief static shots.
2. **Analyze** — librosa extracts BPM, beat times, onset (transient) times, an energy curve, and (via harmonic/percussive source separation) separate melodic-note and drum-hit onset streams from the song.
3. **Pace** — with `--reference`, PySceneDetect (`ContentDetector`, tuned via `--scene-threshold`) finds the reference's cuts, expresses them in beat units at the reference's own tempo, rescales them to the new song's tempo, and snaps every cut to the new song's nearest beat/onset. Without a reference, a heuristic cuts every beat and densifies to onset-level in high-energy sections. `--chorus`/`--drum-buildup`/`--auto-chorus` then override specific time ranges to cut on every melodic note or every drum hit instead.
4. **Assemble** — every cut boundary is snapped to the output's frame grid (using each boundary's own absolute time, not a running sum) before clips are assigned, so rendering can't round segment durations in a way that compounds into audio/video drift across a run of many short cuts. Each segment then gets a source clip (avoiding immediate repeats) and a varied in-point.
5. **Render** — one ffmpeg pass: trim/scale/crop each segment to 1080x1920, hard-cut concat, song muxed as the only audio, H.264/AAC/`+faststart`.

## Tests

```bash
pytest                # fast unit tests, no media involved
pytest -m e2e         # end-to-end: synthesizes test media with ffmpeg, renders a real mp4
```

The yt-dlp download path is not covered by automated tests (live network).
Manual smoke test: put one short YouTube URL in a file and run with
`--urls` and `--dry-run`.
