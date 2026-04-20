#!/usr/bin/env python3
"""Standalone CRT/TBS signal diagnostics.

This does not simulate Topstep execution. It measures whether raw CRT/TBS
setups have directional follow-through over a fixed forward horizon.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_intraday_v3.features.build import build_features


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_bars(path: Path, hdf_key: str) -> pd.DataFrame:
    bars = pd.read_hdf(path, key=hdf_key)
    if "timestamp" in bars.columns:
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        bars = bars.set_index("timestamp")
    elif not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("Bars must have a timestamp column or DatetimeIndex")
    elif bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC")
    else:
        bars.index = bars.index.tz_convert("UTC")
    return bars.sort_index()


def _summarize_signal(work: pd.DataFrame) -> dict:
    if work.empty:
        return {
            "setups": 0,
            "long_setups": 0,
            "short_setups": 0,
            "win_rate": None,
            "avg_forward_return_bps": None,
            "median_forward_return_bps": None,
        }

    directional_return = work["signal"] * work["forward_return"]
    return {
        "setups": int(len(work)),
        "long_setups": int((work["signal"] > 0).sum()),
        "short_setups": int((work["signal"] < 0).sum()),
        "win_rate": float((directional_return > 0).mean()),
        "avg_forward_return_bps": float(directional_return.mean() * 10000.0),
        "median_forward_return_bps": float(directional_return.median() * 10000.0),
        "p25_forward_return_bps": float(np.percentile(directional_return, 25) * 10000.0),
        "p75_forward_return_bps": float(np.percentile(directional_return, 75) * 10000.0),
    }


def run_diagnostics(
    *,
    data_path: Path,
    hdf_key: str,
    feature_cfg_path: Path,
    output_path: Path,
    bar_size: str,
    horizon_bars: int,
) -> dict:
    bars = _load_bars(data_path, hdf_key)
    feature_cfg = _load_yaml(feature_cfg_path)
    features = build_features(bars, bar_size, feature_cfg)

    close = bars["close"].astype(float)
    forward_return = close.shift(-horizon_bars) / close - 1.0
    signal = features["tbs_reclaim_confirmed"].reindex(bars.index).fillna(0).astype(int)

    work = pd.DataFrame(
        {
            "signal": signal,
            "forward_return": forward_return,
            "tbs_sweep_distance_atr": features["tbs_sweep_distance_atr"].reindex(bars.index),
        },
        index=bars.index,
    )
    work = work[(work["signal"] != 0) & work["forward_return"].notna()].copy()
    work["month"] = work.index.tz_convert("America/Chicago").strftime("%Y-%m")

    by_month = {
        str(month): _summarize_signal(group.drop(columns=["month"]))
        for month, group in work.groupby("month", sort=True)
    }
    result = {
        "data_path": str(data_path.relative_to(PROJECT_ROOT) if data_path.is_relative_to(PROJECT_ROOT) else data_path),
        "features_config": str(
            feature_cfg_path.relative_to(PROJECT_ROOT)
            if feature_cfg_path.is_relative_to(PROJECT_ROOT)
            else feature_cfg_path
        ),
        "bar_size": bar_size,
        "horizon_bars": int(horizon_bars),
        "overall": _summarize_signal(work.drop(columns=["month"]) if not work.empty else work),
        "by_month": by_month,
        "promotion_note": (
            "Use this only as raw signal evidence. Promotion still requires the "
            "full CRT/TBS viability A/B or a proper execution backtest."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone CRT/TBS signal diagnostics")
    parser.add_argument("--data-path", default="data/processed/mes_bars_databento_rth.h5")
    parser.add_argument("--hdf-key", default="bars_5min")
    parser.add_argument("--features-config", default="ml_intraday_v3/configs/features_crt_tbs.yaml")
    parser.add_argument("--bar-size", default="5m", choices=["1m", "5m"])
    parser.add_argument("--horizon-bars", type=int, default=6)
    parser.add_argument(
        "--output-path",
        default="ml_intraday_v3/experiments/results/crt_tbs_signal_diagnostics.json",
    )
    args = parser.parse_args()

    result = run_diagnostics(
        data_path=PROJECT_ROOT / args.data_path,
        hdf_key=args.hdf_key,
        feature_cfg_path=PROJECT_ROOT / args.features_config,
        output_path=PROJECT_ROOT / args.output_path,
        bar_size=args.bar_size,
        horizon_bars=args.horizon_bars,
    )
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
