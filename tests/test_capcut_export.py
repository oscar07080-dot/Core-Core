import os

import pytest

from coreedit import capcut


def test_find_drafts_dir_returns_existing_candidate(tmp_path, monkeypatch):
    fake = tmp_path / "CapCut" / "User Data" / "Projects" / "com.lveditor.draft"
    fake.mkdir(parents=True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert capcut.find_drafts_dir() == str(fake)


def test_find_drafts_dir_none_when_nothing_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "nope"))
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path / "home"))
    assert capcut.find_drafts_dir() is None


def test_macos_candidate_checked(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    fake = tmp_path / "Movies" / "CapCut" / "User Data" / "Projects" / "com.lveditor.draft"
    fake.mkdir(parents=True)
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path))
    assert capcut.find_drafts_dir() == str(fake)


def test_export_errors_clearly_without_pyjianyingdraft(tmp_path, monkeypatch):
    monkeypatch.setattr(capcut, "_pyjyd", lambda: None)
    with pytest.raises(RuntimeError, match="pyJianYingDraft"):
        capcut.export_capcut_draft([], object(), str(tmp_path), "draft")


def test_cli_flag_default_and_const():
    from coreedit.cli import build_parser

    p = build_parser()
    base = ["--song", "s.mp3", "--clips-dir", "d"]
    assert p.parse_args(base).export_capcut is None
    assert p.parse_args(base + ["--export-capcut"]).export_capcut == ""
    assert p.parse_args(base + ["--export-capcut", "my_edit"]).export_capcut == "my_edit"
    assert p.parse_args(base + ["--capcut-drafts-dir", "/x"]).capcut_drafts_dir == "/x"
