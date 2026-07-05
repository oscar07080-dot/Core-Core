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

### Key flags

| flag | default | meaning |
|---|---|---|
| `--duration` | 20 | edit length in seconds |
| `--song-start` | 0 | offset into the song (pick the good part) |
| `--reference` | — | edit video whose cut rhythm is copied |
| `--intensity` | medium | cut density (`low`/`medium`/`high`) when no reference |
| `--seed` | — | reproducible clip selection |
| `--source-margin` | 0.5 | seconds skipped at each clip's start/end |
| `--no-repeat-window` | 3 | recent clips excluded from reuse |
| `--output` | `{subject}_{timestamp}.mp4` | output path |

## How it works

1. **Acquire** — yt-dlp downloads (section-only when trimmed) or local globbing; still images (`jpg/png/webp`) are valid sources and become brief static shots.
2. **Analyze** — librosa extracts BPM, beat times, onset (transient) times, and an energy curve from the song.
3. **Pace** — with `--reference`, PySceneDetect finds the reference's cuts, expresses them in beat units at the reference's own tempo, rescales them to the new song's tempo, and snaps every cut to the new song's nearest beat/onset. Without a reference, a heuristic cuts every beat and densifies to onset-level in high-energy sections.
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
