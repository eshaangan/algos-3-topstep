"""Leakage-safe logistic meta-gate for low-frequency event strategies.

This model does not predict trade direction. It uses only information available
before each event to decide whether the event is high confidence. The default
features are lagged strategy outcomes and calendar terms; optional externally
prepared pre-event features may be included as extra numeric columns.

The script uses an expanding walk-forward fit. A threshold is selected only on
predictions before ``--selection-end`` and is then reported on later events.
Results are research diagnostics, not a deployment decision: the final window in
this repository has already been inspected and is not a pristine holdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rule_based_v1.diagnostics.portfolio_mc import AccountRules, loss_floor


def build_features(events: pd.DataFrame) -> pd.DataFrame:
    events = events.sort_values("day").reset_index(drop=True)
    pnl = events["pnl"].astype(float)
    x = pd.DataFrame(index=events.index)
    for window in (4, 8, 13, 26, 52):
        min_n = max(3, window // 2)
        lag = pnl.shift(1)
        x[f"pnl_mean_{window}"] = lag.rolling(window, min_periods=min_n).mean()
        x[f"pnl_std_{window}"] = lag.rolling(window, min_periods=min_n).std()
        x[f"win_rate_{window}"] = (lag > 0).rolling(window, min_periods=min_n).mean()
    x["last_pnl"] = pnl.shift(1)
    if "mae" in events:
        x["last_mae"] = events["mae"].shift(1)
        x["mae_mean_13"] = events["mae"].shift(1).rolling(13, min_periods=6).mean()
    elif "mae_usd" in events:
        x["last_mae"] = events["mae_usd"].shift(1)
        x["mae_mean_13"] = events["mae_usd"].shift(1).rolling(13, min_periods=6).mean()

    prior_equity = pnl.shift(1).fillna(0).cumsum()
    x["strategy_drawdown"] = prior_equity - prior_equity.cummax()
    month = events["day"].dt.month
    x["month_sin"] = np.sin(2 * np.pi * month / 12)
    x["month_cos"] = np.cos(2 * np.pi * month / 12)
    x["quarter_end_month"] = month.isin((3, 6, 9, 12)).astype(float)

    reserved = {"day", "pnl", "mae", "mae_usd"}
    for col in events.columns:
        if col not in reserved and pd.api.types.is_numeric_dtype(events[col]):
            x[f"external_{col}"] = events[col].shift(1)
    return x


def walk_forward_probabilities(events: pd.DataFrame, *, min_train: int = 104, c: float = 0.10) -> tuple[pd.DataFrame, list[str]]:
    events = events.sort_values("day").reset_index(drop=True).copy()
    x = build_features(events)
    y = (events["pnl"] > 0).astype(int)
    valid = x.notna().all(axis=1)
    first_valid = int(np.flatnonzero(valid.to_numpy())[0])
    first_test = max(min_train, first_valid + 80)
    prob = np.full(len(events), np.nan)

    for i in range(first_test, len(events)):
        train_idx = np.arange(first_valid, i)
        train_idx = train_idx[valid.iloc[train_idx].to_numpy()]
        if len(train_idx) < 80 or y.iloc[train_idx].nunique() < 2 or not valid.iloc[i]:
            continue
        model = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=c, max_iter=2_000)),
        ])
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        prob[i] = model.predict_proba(x.iloc[[i]])[0, 1]

    out = events.copy()
    out["prob_win"] = prob
    return out.dropna(subset=["prob_win"]).reset_index(drop=True), list(x.columns)


def choose_threshold(dev: pd.DataFrame, grid: list[float]) -> float:
    """Train-only choice: maximize PnL per calendar event, requiring >=40% coverage."""
    best = grid[0]
    best_score = float("-inf")
    for threshold in grid:
        take = dev["prob_win"] >= threshold
        if take.mean() < 0.40:
            continue
        score = float(dev["pnl"].where(take, 0.0).mean())
        if score > best_score:
            best_score = score
            best = threshold
    return best


def summarize(frame: pd.DataFrame, threshold: float) -> dict:
    y = (frame["pnl"] > 0).astype(int)
    take = frame["prob_win"] >= threshold
    selected = frame.loc[take, "pnl"]
    base = frame["pnl"]
    auc = float(roc_auc_score(y, frame["prob_win"])) if y.nunique() == 2 else None
    return {
        "n_events": int(len(frame)),
        "n_taken": int(take.sum()),
        "coverage": float(take.mean()),
        "auc": auc,
        "brier": float(brier_score_loss(y, frame["prob_win"])),
        "base_total_pnl": float(base.sum()),
        "base_mean_per_event": float(base.mean()),
        "gate_total_pnl": float(selected.sum()),
        "gate_mean_when_taken": float(selected.mean()) if len(selected) else None,
        "gate_mean_per_calendar_event": float(frame["pnl"].where(take, 0.0).mean()),
        "gate_win_rate": float((selected > 0).mean()) if len(selected) else None,
    }


def _paired_block_bootstrap(frame: pd.DataFrame, *, n_paths: int, periods: int, block: int, seed: int) -> np.ndarray:
    values = frame[["pnl", "prob_win"]].to_numpy(float)
    if len(values) < block:
        raise ValueError(f"need at least {block} evaluation events for Monte Carlo")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(periods / block))
    starts = rng.integers(0, len(values) - block + 1, size=(n_paths, n_blocks))
    out = np.empty((n_paths, n_blocks * block, 2), dtype=float)
    offsets = np.arange(block)
    for b in range(n_blocks):
        out[:, b * block : (b + 1) * block, :] = values[starts[:, b, None] + offsets]
    return out[:, :periods, :]


def _run_scaled_path(path: np.ndarray, threshold: float, low_scale: float) -> tuple[str, int]:
    rules = AccountRules()
    balance = peak = 0.0
    best = float("-inf")
    for period, (raw_pnl, prob) in enumerate(path, start=1):
        scale = 1.0 if prob >= threshold else low_scale
        pnl = float(raw_pnl) * scale
        balance += pnl
        peak = max(peak, balance)
        best = max(best, pnl)
        if balance <= loss_floor(peak, rules):
            return "bust", period
        if balance >= rules.target and best <= rules.consistency_fraction * balance:
            return "pass", period
    return "timeout", period


def monte_carlo_comparison(frame: pd.DataFrame, threshold: float, *, n_paths: int, periods: int, block: int, seed: int) -> list[dict]:
    paths = _paired_block_bootstrap(frame, n_paths=n_paths, periods=periods, block=block, seed=seed)
    policies = [
        ("always_2", 1.0, -1.0),
        ("meta_1_or_2", 0.5, threshold),
        ("meta_skip_or_2", 0.0, threshold),
    ]
    rows = []
    for name, low_scale, th in policies:
        effective_threshold = th if th >= 0 else -np.inf
        results = [_run_scaled_path(path, effective_threshold, low_scale) for path in paths]
        status = np.asarray([r[0] for r in results])
        durations = np.asarray([r[1] for r in results])
        passed = durations[status == "pass"]
        rows.append({
            "policy": name,
            "p_pass": float(np.mean(status == "pass")),
            "p_bust": float(np.mean(status == "bust")),
            "median_periods_to_pass": float(np.median(passed)) if len(passed) else None,
        })
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, default=root / "data/processed/weekend_hold_strict.csv")
    ap.add_argument("--selection-end", default="2024-12-31")
    ap.add_argument("--threshold-grid", default="0.50,0.55,0.60,0.65")
    ap.add_argument("--min-train", type=int, default=104)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--mc-paths", type=int, default=20_000)
    ap.add_argument("--mc-periods", type=int, default=260)
    ap.add_argument("--mc-block", type=int, default=13)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    events = pd.read_csv(args.events)
    events["day"] = pd.to_datetime(events["day"], errors="raise")
    predictions, feature_names = walk_forward_probabilities(events, min_train=args.min_train)
    cutoff = pd.Timestamp(args.selection_end)
    dev = predictions[predictions["day"] <= cutoff]
    test = predictions[predictions["day"] > cutoff]
    if dev.empty or test.empty:
        raise ValueError("selection_end must leave non-empty development and evaluation windows")

    grid = [float(v) for v in args.threshold_grid.split(",")]
    threshold = choose_threshold(dev, grid)
    result = {
        "warning": "Post-hoc diagnostic; the evaluation dates have already been inspected in this project.",
        "features": feature_names,
        "selected_threshold": threshold,
        "selection_window": {"start": str(dev.day.min().date()), "end": str(dev.day.max().date())},
        "evaluation_window": {"start": str(test.day.min().date()), "end": str(test.day.max().date())},
        "development": summarize(dev, threshold),
        "evaluation": summarize(test, threshold),
        "evaluation_mc": monte_carlo_comparison(
            test,
            threshold,
            n_paths=args.mc_paths,
            periods=args.mc_periods,
            block=args.mc_block,
            seed=args.seed,
        ),
    }
    print(json.dumps(result, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
