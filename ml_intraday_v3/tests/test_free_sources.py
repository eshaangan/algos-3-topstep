"""
Tests for the free non-OHLCV data fetchers.

Network calls are not made here. Tests that need real data read the on-disk
parquet cache under `data/processed/free/` and skip when it is absent, so the
suite stays runnable offline and in CI.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_intraday_v3.data import free_sources as fs

CACHE = Path("data/processed/free")


def _cached(name: str) -> Path:
    return CACHE / f"{name}.parquet"


def _need(name: str) -> Path:
    p = _cached(name)
    if not p.exists():
        pytest.skip(f"cache {p} absent; run the fetcher once to populate it")
    return p


# ------------------------------------------------------------ pure logic

def test_cboe_index_name_is_validated():
    with pytest.raises(ValueError, match="unknown Cboe index"):
        fs.fetch_cboe_index("NOTANINDEX")


def test_cache_path_creates_directory(tmp_path):
    p = fs._cache_path(tmp_path / "nested", "thing")
    assert p.parent.is_dir()
    assert p.name == "thing.parquet"


def test_module_constants_are_sane():
    assert "VIX" in fs.CBOE_INDICES and "VIX3M" in fs.CBOE_INDICES
    # Coinbase 403s without a browser UA; the header must always be present.
    assert "User-Agent" in fs.UA and fs.UA["User-Agent"]


# ------------------------------------------------- cached-data integrity

def test_vol_complex_has_slopes_and_plausible_levels():
    _need("cboe_vix"), _need("cboe_vix3m"), _need("cboe_vix9d")
    v = fs.fetch_vol_complex()
    assert {"vix", "vix3m", "ts_slope_3m", "ts_slope_9d"} <= set(v.columns)
    assert isinstance(v.index, pd.DatetimeIndex) and v.index.tz is not None
    assert v.index.is_monotonic_increasing
    vix = v["vix"].dropna()
    assert vix.between(5, 200).all(), "VIX outside any plausible historical range"
    # Term structure is usually in contango but must invert during stress.
    s = v["ts_slope_3m"].dropna()
    assert (s > 1).mean() > 0.6
    assert (s < 1).any()


def test_vol_complex_covers_the_weekend_sample():
    _need("cboe_vix3m")
    v = fs.fetch_vol_complex()["ts_slope_3m"].dropna()
    assert v.index.min() <= pd.Timestamp("2020-01-31", tz="UTC")
    assert v.index.max() >= pd.Timestamp("2026-06-30", tz="UTC")


def test_cot_positioning_shape_and_lookahead_safety():
    _need("cot_e-mini_sandp_500")
    c = fs.cot_positioning("E-MINI S&P 500")
    assert {"noncomm_net", "noncomm_net_pct_oi", "noncomm_net_pct_oi_z"} <= set(c.columns)
    assert c.index.is_monotonic_increasing
    # Net position cannot exceed open interest in magnitude.
    ratio = (c["noncomm_net"].abs() / c["oi"]).dropna()
    assert (ratio <= 1.0).all()
    # The z-score uses a shifted window, so the first rows must be NaN.
    assert c["noncomm_net_pct_oi_z"].iloc[0] != c["noncomm_net_pct_oi_z"].iloc[0] \
        or np.isnan(c["noncomm_net_pct_oi_z"].iloc[0])


def test_cot_raises_a_clear_error_on_an_empty_response(monkeypatch, tmp_path):
    """An unknown contract name returns [] from CFTC; that must be a clear error."""
    monkeypatch.setattr(fs, "_get", lambda url, timeout=45: b"[]")
    with pytest.raises(ValueError, match="no COT rows"):
        fs.fetch_cot("DEFINITELY NOT A CONTRACT", cache_dir=tmp_path, refresh=True)


def test_cot_missing_columns_raise_keyerror(monkeypatch, tmp_path):
    """A schema change upstream must fail loudly, not produce silent NaNs."""
    monkeypatch.setattr(
        fs, "fetch_cot",
        lambda *a, **k: pd.DataFrame(
            {"open_interest_all": [1, 2]},
            index=pd.DatetimeIndex(["2024-01-02", "2024-01-09"], tz="UTC"),
        ),
    )
    with pytest.raises(KeyError):
        fs.cot_positioning("E-MINI S&P 500", cache_dir=tmp_path)


def test_coinbase_cache_is_hourly_and_ordered():
    _need("coinbase_btc_usd_1h")
    d = fs.fetch_coinbase_hourly("BTC-USD")
    assert {"open", "high", "low", "close", "volume"} <= set(d.columns)
    assert d.index.is_monotonic_increasing and d.index.is_unique
    assert d.index.tz is not None
    gaps = d.index.to_series().diff().dropna()
    # The modal spacing must be one hour.
    assert gaps.mode().iloc[0] == pd.Timedelta(hours=1)
    assert (d["high"] >= d["low"]).all()


def test_coinbase_covers_the_futures_closure_windows():
    """The weekend test needs Fri 17:00 -> Sun 17:00 ET for 2020-2026."""
    _need("coinbase_btc_usd_1h")
    d = fs.fetch_coinbase_hourly("BTC-USD")
    assert d.index.min() <= pd.Timestamp("2020-01-05", tz="UTC")
    assert d.index.max() >= pd.Timestamp("2026-06-30", tz="UTC")
    # Crypto trades while equity futures are shut: Saturdays must be present.
    et = d.index.tz_convert("America/New_York")
    assert (et.dayofweek == 5).sum() > 300
