"""
Quick diagnostics to verify label <-> execution alignment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data.clean_bars import clean_bars
from features.engineer import add_features
from features.labels_aligned import make_aligned_fixed_horizon_labels
from models.nn_inference import load_nn_bundle


def _sample_alignment(
    bars: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    horizon_bars: int,
    tick_size: float,
    tick_value: float,
    entry_price_col: str,
    exit_price_col: str,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    max_idx = int(labels["idx"].max())
    if max_idx <= 1:
        return pd.DataFrame()
    sample_idx = rng.choice(np.arange(0, max_idx + 1), size=min(samples, max_idx + 1), replace=False)
    rows = []
    for idx in sample_idx:
        entry_idx = idx + 1
        exit_idx = idx + 1 + horizon_bars
        if exit_idx >= len(bars):
            continue
        entry_price = float(bars.iloc[entry_idx][entry_price_col])
        exit_price = float(bars.iloc[exit_idx][exit_price_col])
        ret_ticks = (exit_price - entry_price) / tick_size
        label_row = labels[labels["idx"] == idx].iloc[0]
        label_ret_ticks = float(label_row["ret_ticks"])
        pnl = ret_ticks * tick_value
        rows.append(
            {
                "idx": int(idx),
                "entry_idx": int(entry_idx),
                "exit_idx": int(exit_idx),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "ret_ticks": ret_ticks,
                "label_ret_ticks": label_ret_ticks,
                "ret_diff": ret_ticks - label_ret_ticks,
                "pnl": pnl,
                "y_long": int(label_row["y_long"]),
                "y_short": int(label_row["y_short"]),
            }
        )
    return pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)


def _time_shift_collapse(
    bars: pd.DataFrame, labels: pd.DataFrame, feature_name: str = "returns_1"
) -> Optional[dict]:
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return None

    features = add_features(bars, verbose=False)
    features = features.reset_index(drop=True)
    if "idx" not in features.columns:
        features.insert(0, "idx", np.arange(len(features)))

    merged = labels.merge(features, on="idx", how="inner")
    if feature_name not in merged.columns:
        return None

    scores = merged[feature_name].values
    labels_long = merged["y_long"].values
    valid = np.isfinite(scores)
    if valid.sum() < 10 or len(np.unique(labels_long[valid])) < 2:
        return None

    auc_aligned = float(roc_auc_score(labels_long[valid], scores[valid]))
    shifted_scores = pd.Series(scores).shift(1).values
    valid_shift = np.isfinite(shifted_scores)
    if valid_shift.sum() < 10:
        return None
    auc_shifted = float(roc_auc_score(labels_long[valid_shift], shifted_scores[valid_shift]))
    return {"auc_aligned": auc_aligned, "auc_shifted": auc_shifted}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check aligned label assumptions.")
    parser.add_argument("--data-path", default="data/processed/es_bars_2010_2025.h5")
    parser.add_argument("--model-dir", default="models/nn_saved")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    bundle = load_nn_bundle(args.model_dir, fold=args.fold)
    nn_cfg = bundle.config.get("nn_config", {})

    horizon_bars = int(nn_cfg["horizon_bars"])
    threshold_ticks = int(nn_cfg["threshold_ticks"])
    tick_size = float(nn_cfg["tick_size"])
    tick_value = float(nn_cfg["tick_value"])
    entry_price_col = str(nn_cfg.get("label_entry_price_col", "open"))
    exit_price_col = str(nn_cfg.get("label_exit_price_col", "close"))

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    with pd.HDFStore(data_path, "r") as store:
        bars = store["bars_5min"].copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    bars = clean_bars(bars, tick_size=tick_size, verbose=False)

    labels = make_aligned_fixed_horizon_labels(
        bars,
        horizon_bars=horizon_bars,
        threshold_ticks=threshold_ticks,
        tick_size=tick_size,
        entry_price_col=entry_price_col,
        exit_price_col=exit_price_col,
    )

    print("\nAligned label settings:")
    print(f"  horizon_bars={horizon_bars} | threshold_ticks={threshold_ticks}")
    print(f"  entry={entry_price_col} (t+1) | exit={exit_price_col} (t+1+horizon)")
    print(f"  label_version={nn_cfg.get('label_version')}")

    sample_df = _sample_alignment(
        bars,
        labels,
        horizon_bars=horizon_bars,
        tick_size=tick_size,
        tick_value=tick_value,
        entry_price_col=entry_price_col,
        exit_price_col=exit_price_col,
        samples=args.samples,
        seed=args.seed,
    )
    print("\nSample aligned labels:")
    print(sample_df.to_string(index=False))

    collapse = _time_shift_collapse(bars, labels)
    if collapse:
        print("\nTime-shift collapse check (feature=returns_1):")
        print(f"  AUC aligned: {collapse['auc_aligned']:.3f}")
        print(f"  AUC shifted: {collapse['auc_shifted']:.3f}")
        if abs(collapse["auc_aligned"] - collapse["auc_shifted"]) < 0.02:
            print("  WARNING: AUC did not collapse after shift; verify label alignment.")
    else:
        print("\nTime-shift collapse check: skipped (insufficient data or sklearn missing).")


if __name__ == "__main__":
    main()
