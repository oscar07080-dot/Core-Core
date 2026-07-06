# Core-Core

CLI tool that generates TikTok-ready, beat-synced montage edits ("core edits"):
hard cuts on the song's beat, near-strobe cutting in high-energy sections,
vertical 1080x1920 output with the song as the only audio.

## Setup

Requires Python 3.10+ and the `ffmpeg`/`ffprobe` binaries on your PATH.

```bash
pip install -r requirements.txt
```

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
```

### URL list format (`urls.txt`)

One source per line: `URL|START|END` (start/end in seconds, both optional —
when given, only that section is downloaded). `#` starts a comment.

```
https://www.youtube.com/watch?v=XXXXXXXXXXX
https://www.youtube.com/watch?v=YYYYYYYYYYY|90|120
# a comment
```

Downloads are cached in `--cache-dir` (default `./cache`); re-runs reuse them.

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

Cuts inside these ranges aren't a metronomically constant rate: each
note/hit's actual strength decides whether it gets its own cut or merges
into the previous one, so accented notes reliably cut while quiet ones
often don't. Tune this with `--section-variation` (0 = only accents cut,
maximum variation; 1 = uniform, every single note/hit cuts; default 0.35).

### Key flags

| flag | default | meaning |
|---|---|---|
| `--duration` | 20 | edit length in seconds |
| `--song-start` | 0 | offset into the song (pick the good part) |
| `--reference` | — | edit video whose cut rhythm is copied |
| `--scene-threshold` | 12.0 | cut-detection sensitivity for `--reference` (lower = catches more subtle cuts, risks false positives) |
| `--intensity` | medium | cut density (`low`/`medium`/`high`) when no reference |
| `--chorus` | — | `START:END` range to cut on every melodic note (repeatable) |
| `--drum-buildup` | — | `START:END` range to cut on every drum hit (repeatable) |
| `--auto-chorus` | off | auto-detect the chorus/build-up instead of specifying ranges |
| `--section-variation` | 0.35 | pacing variation within chorus/build-up ranges (0=accents only, 1=uniform) |
| `--seed` | — | reproducible clip selection |
| `--source-margin` | 0.5 | seconds skipped at each clip's start/end |
| `--no-repeat-window` | 3 | recent clips excluded from reuse |
| `--output` | `{subject}_{timestamp}.mp4` | output path |

## How it works

1. **Acquire** — yt-dlp downloads (section-only when trimmed) or local globbing; still images (`jpg/png/webp`) are valid sources and become brief static shots.
2. **Analyze** — librosa extracts BPM, beat times, onset (transient) times, an energy curve, and (via harmonic/percussive source separation) separate melodic-note and drum-hit onset streams from the song.
3. **Pace** — with `--reference`, PySceneDetect (`ContentDetector`, tuned via `--scene-threshold`) finds the reference's cuts, expresses them in beat units at the reference's own tempo, rescales them to the new song's tempo, and snaps every cut to the new song's nearest beat/onset. Without a reference, a heuristic cuts every beat and densifies to onset-level in high-energy sections. `--chorus`/`--drum-buildup`/`--auto-chorus` then override specific time ranges to cut on every melodic note or every drum hit instead.
4. **Assemble** — each segment gets a source clip (avoiding immediate repeats) and a varied in-point.
5. **Render** — one ffmpeg pass: trim/scale/crop each segment to 1080x1920, hard-cut concat, song muxed as the only audio, H.264/AAC/`+faststart`.

## Tests

```bash
pytest                # fast unit tests, no media involved
pytest -m e2e         # end-to-end: synthesizes test media with ffmpeg, renders a real mp4
```

The yt-dlp download path is not covered by automated tests (live network).
Manual smoke test: put one short YouTube URL in a file and run with
`--urls` and `--dry-run`.
