"""Export the computed timeline as a native CapCut desktop draft (project).

Writes a draft folder -- draft_content.json, draft_meta_info.json, and all
media -- directly into CapCut's drafts directory, so the project shows up on
CapCut's home screen with the whole timeline pre-built: one pre-trimmed clip
per cut, the song on the audio track. The user then only has to Replace each
clip with their own footage.

The draft format is undocumented and reverse-engineered (via the community
pyJianYingDraft library), so this is best-effort: it targets the plain-JSON
draft format used by CapCut desktop / JianYing <= 5.9. If a given CapCut
version refuses the draft, --export-clips remains the always-works fallback.

The media files are rendered *inside* the draft folder and referenced by
absolute path, which is why the draft must be generated straight into the
real drafts directory on the machine that runs CapCut -- moving a draft
folder afterwards would break the absolute paths recorded in the JSON.
"""

from __future__ import annotations

import os

from .models import EditSpec, Segment
from .render import render_clip_sequence

INSTALL_HINT = (
    "the CapCut draft exporter needs the pyJianYingDraft package:\n"
    "  pip install pyJianYingDraft"
)

# Standard CapCut/JianYing desktop draft locations. %LOCALAPPDATA% entries
# are Windows, ~/Movies entries are macOS; JianyingPro is the Chinese build
# of the same editor and uses the same draft format.
_WINDOWS_SUBPATH = os.path.join("User Data", "Projects", "com.lveditor.draft")


def _candidate_drafts_dirs() -> list[str]:
    home = os.path.expanduser("~")
    candidates: list[str] = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates += [
            os.path.join(local, "CapCut", _WINDOWS_SUBPATH),
            os.path.join(local, "JianyingPro", _WINDOWS_SUBPATH),
        ]
    candidates += [
        os.path.join(home, "Movies", "CapCut", _WINDOWS_SUBPATH),
        os.path.join(home, "Movies", "JianyingPro", _WINDOWS_SUBPATH),
    ]
    return candidates


def find_drafts_dir() -> str | None:
    """Locate CapCut desktop's drafts directory on this machine, if any."""
    for path in _candidate_drafts_dirs():
        if os.path.isdir(path):
            return path
    return None


def _pyjyd():
    try:
        import pyJianYingDraft

        return pyJianYingDraft
    except ImportError:
        return None


def export_capcut_draft(
    segments: list[Segment],
    spec: EditSpec,
    drafts_dir: str,
    draft_name: str,
    verbose: bool = False,
) -> str:
    """Build a CapCut draft named `draft_name` inside `drafts_dir`.

    Renders every segment as a numbered pre-trimmed clip (plus the trimmed
    song) into the draft's own media/ subfolder, then lays them out
    back-to-back on a video track with the song on an audio track.

    Returns the created draft folder path.
    """
    pyjyd = _pyjyd()
    if pyjyd is None:
        raise RuntimeError(INSTALL_HINT)

    # create_draft writes draft_meta_info.json and gives us a ScriptFile
    # whose save() targets <drafts_dir>/<draft_name>/draft_content.json
    script = pyjyd.DraftFolder(drafts_dir).create_draft(
        draft_name, spec.width, spec.height, fps=spec.fps
    )
    draft_dir = os.path.join(drafts_dir, draft_name)
    media_dir = os.path.join(draft_dir, "media")
    clip_paths = render_clip_sequence(segments, spec, media_dir, verbose=verbose)

    script.append_track(pyjyd.TrackSpec(pyjyd.TrackType.video, name="clips"))
    script.append_track(pyjyd.TrackSpec(pyjyd.TrackType.audio, name="song"))

    sec = pyjyd.SEC
    start_us = 0
    for seg, path in zip(segments, clip_paths):
        dur_us = int(round(seg.duration * sec))
        material = pyjyd.VideoMaterial(path)
        # the encoded clip can come out a frame short of the requested
        # duration; a source range past the material's end would be invalid
        src_dur = min(dur_us, material.duration)
        script.add_segment(
            pyjyd.VideoSegment(
                material,
                pyjyd.Timerange(start_us, dur_us),
                source_timerange=pyjyd.Timerange(0, src_dur),
            ),
            "clips",
        )
        start_us += dur_us

    song_material = pyjyd.AudioMaterial(os.path.join(media_dir, "song.m4a"))
    audio_dur = min(start_us, song_material.duration)
    script.add_segment(
        pyjyd.AudioSegment(
            song_material,
            pyjyd.Timerange(0, audio_dur),
            source_timerange=pyjyd.Timerange(0, audio_dur),
        ),
        "song",
    )
    script.save()
    return draft_dir
