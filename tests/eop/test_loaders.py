from pathlib import Path

import jax.numpy as jnp
import pytest

from difforb.core.constants import MJD_START
from difforb.core.eop.container import EarthOrientationData
from difforb.core.eop import loaders as eop_loaders
from difforb.core.eop.loaders import load_iers_eopc04, parse_iers_eopc04
from tests.assertions import assert_allclose


def test_parse_iers_eopc04_parses_selected_rows(eopc04_sample_path):
    eop = parse_iers_eopc04(str(eopc04_sample_path))

    assert isinstance(eop, EarthOrientationData)

    assert_allclose(eop.xpoles[0], -0.012700, atol=0.0, rtol=0.0)
    assert_allclose(eop.ypoles[0], 0.213000, atol=0.0, rtol=0.0)
    assert_allclose(eop.ut1dutcs[0], 0.0326338, atol=0.0, rtol=0.0)
    assert_allclose(eop.dpsis[0], 0.0, atol=0.0, rtol=0.0)
    assert_allclose(eop.depss[0], 0.0, atol=0.0, rtol=0.0)

    assert_allclose(eop.xpoles[3], -0.021999, atol=0.0, rtol=0.0)
    assert_allclose(eop.ypoles[3], 0.216301, atol=0.0, rtol=0.0)
    assert_allclose(eop.ut1dutcs[3], 0.0311435, atol=0.0, rtol=0.0)
    assert_allclose(eop.dpsis[3], 0.0, atol=0.0, rtol=0.0)
    assert_allclose(eop.depss[3], 0.0, atol=0.0, rtol=0.0)

    assert_allclose(eop.xpoles[-1], -0.027599, atol=0.0, rtol=0.0)
    assert_allclose(eop.ypoles[-1], 0.218301, atol=0.0, rtol=0.0)
    assert_allclose(eop.ut1dutcs[-1], 0.0305353, atol=0.0, rtol=0.0)
    assert_allclose(eop.dpsis[-1], 0.0, atol=0.0, rtol=0.0)
    assert_allclose(eop.depss[-1], 0.0, atol=0.0, rtol=0.0)


def test_parse_iers_eopc04_sets_final_date_range_from_table_edges(eopc04_sample_path):
    eop = parse_iers_eopc04(str(eopc04_sample_path))

    expected = jnp.asarray([37665.0 + MJD_START, 37670.0 + MJD_START])

    assert_allclose(eop.final_date_range, expected, atol=0.0, rtol=0.0)


def test_parse_iers_eopc04_returns_expected_container_contract(eopc04_sample_path):
    eop = parse_iers_eopc04(str(eopc04_sample_path))

    assert isinstance(eop, EarthOrientationData)
    assert eop.predicted_date_range is None

    assert eop.tt_jds.shape == (6,)
    assert eop.tt_jds.shape == eop.xpoles.shape
    assert eop.tt_jds.shape == eop.ypoles.shape
    assert eop.tt_jds.shape == eop.ut1dutcs.shape
    assert eop.tt_jds.shape == eop.dpsis.shape
    assert eop.tt_jds.shape == eop.depss.shape


def test_load_iers_eopc04_auto_update_false_uses_local_file_without_download(
    eopc04_sample_path,
    monkeypatch,
):
    download_called = False

    def _unexpected_download(*_args, **_kwargs):
        nonlocal download_called
        download_called = True
        raise AssertionError("load_iers_eopc04(auto_update=False) should not download when the local file exists")

    monkeypatch.setattr(eop_loaders, "DEFAULT_IERS_EOP_C04_FILENAME", str(eopc04_sample_path))
    monkeypatch.setattr(eop_loaders, "download", _unexpected_download)

    loaded = load_iers_eopc04(auto_update=False)
    parsed = parse_iers_eopc04(str(eopc04_sample_path))

    assert download_called is False
    assert isinstance(loaded, EarthOrientationData)
    assert_allclose(loaded.tt_jds, parsed.tt_jds, atol=0.0, rtol=0.0)
    assert_allclose(loaded.xpoles, parsed.xpoles, atol=0.0, rtol=0.0)
    assert_allclose(loaded.ypoles, parsed.ypoles, atol=0.0, rtol=0.0)
    assert_allclose(loaded.ut1dutcs, parsed.ut1dutcs, atol=0.0, rtol=0.0)
    assert_allclose(loaded.dpsis, parsed.dpsis, atol=0.0, rtol=0.0)
    assert_allclose(loaded.depss, parsed.depss, atol=0.0, rtol=0.0)
    assert_allclose(loaded.final_date_range, parsed.final_date_range, atol=0.0, rtol=0.0)


def test_load_iers_eopc04_downloads_when_local_file_is_missing(
    eopc04_sample_path,
    monkeypatch,
    tmp_path,
):
    target_path = tmp_path / "eopc04.dPsi_dEps.1962-now.txt"
    download_calls = []

    def _fake_download(filepath, url):
        download_calls.append((filepath, url))
        Path(filepath).write_text(eopc04_sample_path.read_text())

    monkeypatch.setattr(eop_loaders, "DEFAULT_IERS_EOP_C04_FILENAME", str(target_path))
    monkeypatch.setattr(eop_loaders, "download", _fake_download)

    loaded = load_iers_eopc04(auto_update=False)
    parsed = parse_iers_eopc04(str(eopc04_sample_path))

    assert target_path.exists()
    assert download_calls == [
        (str(target_path), eop_loaders.DEFAULT_IERS_EOP_C04_URL),
    ]
    assert isinstance(loaded, EarthOrientationData)
    assert_allclose(loaded.tt_jds, parsed.tt_jds, atol=0.0, rtol=0.0)
    assert_allclose(loaded.xpoles, parsed.xpoles, atol=0.0, rtol=0.0)
    assert_allclose(loaded.ypoles, parsed.ypoles, atol=0.0, rtol=0.0)
    assert_allclose(loaded.ut1dutcs, parsed.ut1dutcs, atol=0.0, rtol=0.0)
    assert_allclose(loaded.dpsis, parsed.dpsis, atol=0.0, rtol=0.0)
    assert_allclose(loaded.depss, parsed.depss, atol=0.0, rtol=0.0)
    assert_allclose(loaded.final_date_range, parsed.final_date_range, atol=0.0, rtol=0.0)


def test_load_iers_eopc04_auto_update_true_skips_download_when_file_is_recent(
    eopc04_sample_path,
    monkeypatch,
):
    from difforb.core.time import utils as time_utils

    download_called = False
    parsed = parse_iers_eopc04(str(eopc04_sample_path))

    def _unexpected_download(*_args, **_kwargs):
        nonlocal download_called
        download_called = True
        raise AssertionError("load_iers_eopc04(auto_update=True) should not download when the local file is recent")

    monkeypatch.setattr(eop_loaders, "DEFAULT_IERS_EOP_C04_FILENAME", str(eopc04_sample_path))
    monkeypatch.setattr(eop_loaders, "download", _unexpected_download)
    monkeypatch.setattr(
        time_utils,
        "julian_date",
        lambda *_args, **_kwargs: (parsed.final_date_range[1] + 35.0, 0.0),
    )

    loaded = load_iers_eopc04(auto_update=True)

    assert download_called is False
    assert isinstance(loaded, EarthOrientationData)
    assert_allclose(loaded.tt_jds, parsed.tt_jds, atol=0.0, rtol=0.0)
    assert_allclose(loaded.xpoles, parsed.xpoles, atol=0.0, rtol=0.0)
    assert_allclose(loaded.ypoles, parsed.ypoles, atol=0.0, rtol=0.0)
    assert_allclose(loaded.ut1dutcs, parsed.ut1dutcs, atol=0.0, rtol=0.0)
    assert_allclose(loaded.dpsis, parsed.dpsis, atol=0.0, rtol=0.0)
    assert_allclose(loaded.depss, parsed.depss, atol=0.0, rtol=0.0)
    assert_allclose(loaded.final_date_range, parsed.final_date_range, atol=0.0, rtol=0.0)


def test_load_iers_eopc04_auto_update_true_downloads_when_file_is_stale(
    eopc04_sample_path,
    monkeypatch,
    tmp_path,
):
    from difforb.core.time import utils as time_utils

    target_path = tmp_path / "eopc04.dPsi_dEps.1962-now.txt"
    target_path.write_text(eopc04_sample_path.read_text())

    stale = parse_iers_eopc04(str(target_path))
    refreshed_text = eopc04_sample_path.read_text().replace("-0.012700", "-0.112700", 1)
    download_calls = []

    def _fake_download(filepath, url):
        download_calls.append((filepath, url))
        Path(filepath).write_text(refreshed_text)

    monkeypatch.setattr(eop_loaders, "DEFAULT_IERS_EOP_C04_FILENAME", str(target_path))
    monkeypatch.setattr(eop_loaders, "download", _fake_download)
    monkeypatch.setattr(
        time_utils,
        "julian_date",
        lambda *_args, **_kwargs: (stale.final_date_range[1] + 35.1, 0.0),
    )

    loaded = load_iers_eopc04(auto_update=True)
    refreshed = parse_iers_eopc04(str(target_path))

    assert download_calls == [
        (str(target_path), eop_loaders.DEFAULT_IERS_EOP_C04_URL),
    ]
    assert isinstance(loaded, EarthOrientationData)
    assert_allclose(loaded.tt_jds, refreshed.tt_jds, atol=0.0, rtol=0.0)
    assert_allclose(loaded.xpoles, refreshed.xpoles, atol=0.0, rtol=0.0)
    assert_allclose(loaded.ypoles, refreshed.ypoles, atol=0.0, rtol=0.0)
    assert_allclose(loaded.ut1dutcs, refreshed.ut1dutcs, atol=0.0, rtol=0.0)
    assert_allclose(loaded.dpsis, refreshed.dpsis, atol=0.0, rtol=0.0)
    assert_allclose(loaded.depss, refreshed.depss, atol=0.0, rtol=0.0)
    assert_allclose(loaded.final_date_range, refreshed.final_date_range, atol=0.0, rtol=0.0)
    assert_allclose(loaded.xpoles[0], -0.112700, atol=0.0, rtol=0.0)


def test_load_iers_eopc04_uses_fallback_url_when_primary_fails(
    eopc04_sample_path,
    monkeypatch,
    tmp_path,
):
    target_path = tmp_path / "eopc04.dPsi_dEps.1962-now.txt"
    download_calls = []

    def _fake_download(filepath, url):
        download_calls.append((filepath, url))
        if url == "primary":
            raise OSError("primary unavailable")
        Path(filepath).write_text(eopc04_sample_path.read_text())

    monkeypatch.setattr(eop_loaders, "DEFAULT_IERS_EOP_C04_FILENAME", str(target_path))
    monkeypatch.setattr(eop_loaders, "DEFAULT_IERS_EOP_C04_URLS", ("primary", "fallback"))
    monkeypatch.setattr(eop_loaders, "download", _fake_download)

    loaded = load_iers_eopc04(auto_update=False)
    parsed = parse_iers_eopc04(str(eopc04_sample_path))

    assert target_path.exists()
    assert download_calls == [
        (str(target_path), "primary"),
        (str(target_path), "fallback"),
    ]
    assert isinstance(loaded, EarthOrientationData)
    assert_allclose(loaded.tt_jds, parsed.tt_jds, atol=0.0, rtol=0.0)
    assert_allclose(loaded.xpoles, parsed.xpoles, atol=0.0, rtol=0.0)
    assert_allclose(loaded.ypoles, parsed.ypoles, atol=0.0, rtol=0.0)
    assert_allclose(loaded.ut1dutcs, parsed.ut1dutcs, atol=0.0, rtol=0.0)
    assert_allclose(loaded.dpsis, parsed.dpsis, atol=0.0, rtol=0.0)
    assert_allclose(loaded.depss, parsed.depss, atol=0.0, rtol=0.0)
    assert_allclose(loaded.final_date_range, parsed.final_date_range, atol=0.0, rtol=0.0)
