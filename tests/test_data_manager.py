import pytest

from difforb.data import manager


class _Response:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


def _dataset_entry(path: str) -> dict:
    return {
        "installable": True,
        "files": [path],
        "sources": [
            {"path": path, "url": "primary"},
            {"path": path, "url": "fallback"},
        ],
    }


def test_install_dataset_uses_fallback_source_for_same_target(monkeypatch, tmp_path):
    path = "iers/eopc04.dPsi_dEps.1962-now.txt"
    calls = []

    def _fake_get(url, timeout):
        calls.append((url, timeout))
        if url == "primary":
            raise TimeoutError("primary unavailable")
        return _Response(b"fallback")

    monkeypatch.setattr(manager, "get_user_data_dir", lambda: tmp_path)
    monkeypatch.setattr(manager, "get_dataset", lambda _name: _dataset_entry(path))
    monkeypatch.setattr(manager.requests, "get", _fake_get)

    written = manager.install_dataset("eop")

    target = tmp_path / path
    assert written == (target,)
    assert target.read_bytes() == b"fallback"
    assert calls == [("primary", 60), ("fallback", 60)]


def test_install_dataset_does_not_overwrite_primary_with_fallback_when_forced(monkeypatch, tmp_path):
    path = "iers/eopc04.dPsi_dEps.1962-now.txt"
    calls = []

    def _fake_get(url, timeout):
        calls.append((url, timeout))
        return _Response(url.encode())

    monkeypatch.setattr(manager, "get_user_data_dir", lambda: tmp_path)
    monkeypatch.setattr(manager, "get_dataset", lambda _name: _dataset_entry(path))
    monkeypatch.setattr(manager.requests, "get", _fake_get)

    written = manager.install_dataset("eop", force=True)

    target = tmp_path / path
    assert written == (target,)
    assert target.read_bytes() == b"primary"
    assert calls == [("primary", 60)]


def test_install_dataset_force_raises_when_refresh_sources_fail(monkeypatch, tmp_path):
    path = "iers/eopc04.dPsi_dEps.1962-now.txt"
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old")
    calls = []

    def _fake_get(url, timeout):
        calls.append((url, timeout))
        raise TimeoutError(f"{url} unavailable")

    monkeypatch.setattr(manager, "get_user_data_dir", lambda: tmp_path)
    monkeypatch.setattr(manager, "get_dataset", lambda _name: _dataset_entry(path))
    monkeypatch.setattr(manager.requests, "get", _fake_get)

    with pytest.raises(manager.DataNotInstalledError, match="Could not refresh"):
        manager.install_dataset("eop", force=True)

    assert target.read_bytes() == b"old"
    assert calls == [("primary", 60), ("fallback", 60)]


def test_install_file_source_extracts_mpc_html_pre(monkeypatch, tmp_path):
    path = "obs_code/iau_obs_codes.txt"
    html = b"""<!doctype html>
<html><body><pre>Code  Long.    cos       sin     Name
000   0.0000  0.62411  +0.77873  Greenwich
568 204.5278  0.94171  +0.33725  <a href="https://example.test">Maunakea</a>
</pre></body></html>
"""

    def _fake_get(url, timeout):
        return _Response(html)

    monkeypatch.setattr(manager, "get_user_data_dir", lambda: tmp_path)
    monkeypatch.setattr(manager.requests, "get", _fake_get)

    written = manager._install_file_source(
        {
            "path": path,
            "url": "mpc-html",
            "transform": "html_pre",
        },
        force=True,
    )

    assert written == tmp_path / path
    assert written.read_text() == (
        "Code  Long.    cos       sin     Name\n"
        "000   0.0000  0.62411  +0.77873  Greenwich\n"
        "568 204.5278  0.94171  +0.33725  Maunakea\n"
    )
