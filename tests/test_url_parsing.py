import pytest

from coreedit.acquisition import UrlEntry, cache_key, parse_url_list, ydl_options
from coreedit.acquisition import _detect_proxy


def test_bare_url():
    entries = parse_url_list("https://example.com/watch?v=abc\n")
    assert entries == [UrlEntry(url="https://example.com/watch?v=abc")]


def test_url_with_start_and_end():
    entries = parse_url_list("https://example.com/v|90|120.5\n")
    assert entries[0].start == 90.0
    assert entries[0].end == 120.5


def test_url_with_start_only():
    entries = parse_url_list("https://example.com/v|30\n")
    assert entries[0].start == 30.0
    assert entries[0].end is None


def test_comments_and_blank_lines_skipped():
    text = "# comment\n\nhttps://example.com/a\n   \n# another\nhttps://example.com/b\n"
    entries = parse_url_list(text)
    assert [e.url for e in entries] == ["https://example.com/a", "https://example.com/b"]


def test_pipe_in_query_string_is_safe():
    # URLs contain ?&= but not |, so pipe-splitting keeps the URL intact
    entries = parse_url_list("https://example.com/watch?v=a&t=5s|10|20\n")
    assert entries[0].url == "https://example.com/watch?v=a&t=5s"


def test_non_url_rejected():
    with pytest.raises(ValueError, match="not a URL"):
        parse_url_list("notaurl.mp4\n")


def test_non_numeric_trim_rejected():
    with pytest.raises(ValueError, match="must be numbers"):
        parse_url_list("https://example.com/v|abc\n")


def test_end_before_start_rejected():
    with pytest.raises(ValueError, match="END must be greater"):
        parse_url_list("https://example.com/v|20|10\n")


def test_too_many_fields_rejected():
    with pytest.raises(ValueError, match="too many fields"):
        parse_url_list("https://example.com/v|1|2|3\n")


def test_cache_key_stable_and_range_sensitive():
    a = UrlEntry(url="https://example.com/v")
    b = UrlEntry(url="https://example.com/v", start=10.0)
    assert cache_key(a) == cache_key(a)
    assert cache_key(a) != cache_key(b)


def test_ydl_options_plain_download_has_no_ranges():
    opts = ydl_options(UrlEntry(url="https://example.com/v"), cache_dir="/tmp/x")
    assert "download_ranges" not in opts
    assert opts["noplaylist"] is True


def test_ydl_options_trimmed_download_sets_ranges():
    opts = ydl_options(UrlEntry(url="https://example.com/v", start=5.0, end=9.0), cache_dir="/tmp/x")
    assert "download_ranges" in opts
    assert opts["force_keyframes_at_cuts"] is True


def test_ydl_options_passes_proxy_when_set(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9999")
    opts = ydl_options(UrlEntry(url="https://example.com/v"), cache_dir="/tmp/x")
    assert opts["proxy"] == "http://127.0.0.1:9999"


def test_ydl_options_no_proxy_key_when_unset(monkeypatch):
    for var in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    opts = ydl_options(UrlEntry(url="https://example.com/v"), cache_dir="/tmp/x")
    assert "proxy" not in opts


def test_detect_proxy_checks_expected_vars_in_order(monkeypatch):
    for var in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    assert _detect_proxy() is None

    monkeypatch.setenv("ALL_PROXY", "http://fallback:8080")
    assert _detect_proxy() == "http://fallback:8080"

    monkeypatch.setenv("HTTPS_PROXY", "http://preferred:8080")
    assert _detect_proxy() == "http://preferred:8080"
