"""
Run Batch 20: Meta-Labeling on Rule-Based Primary
==================================================
Executes a single meta-labeling experiment config (batch20_metalabel_NNN.json).
Designed to run on GCP (one config per VM) or locally for debugging.

The correct meta-labeling architecture (per López de Prado):
  1. Run rule_based_v1 on training data → get primary trade signals + outcomes
  2. For each primary signal, build a feature vector from the N bars leading up to entry
  3. Meta-label: y=1 if the primary signal was profitable, y=0 if not
  4. Train secondary model to predict "should we take this trade?"
  5. CPCV cross-validate the combined (primary + secondary) system
  6. Report: meta AUC, precision before/after filtering, lift %, OOS metrics

Usage (local):
    python ml_intraday_v3/experiments/run_metalabel_rulebased.py \
        --config ml_intraday_v3/experiments/batch20_metalabel_configs/batch20_metalabel_001.json \
        --data-path ml_intraday_v3/data/processed/mes_bars_databento_rth.h5 \
        --output ml_intraday_v3/experiments/results/batch20/

Usage (GCP):
    # Config passed via GCS or local disk after gsutil cp
    python run_metalabel_rulebased.py --config /tmp/batch20_metalabel_001.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent
_RBV1_DIR = _PROJECT_ROOT / "rule_based_v1"

for _p in [str(_PROJECT_ROOT), str(_RBV1_DIR), str(_HERE.parent.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.backtest_engine import BacktestEngine        # noqa: E402
from engine.signal_aggregator import SignalAggregator   # noqa: E402
from engine.risk_manager import RiskManager             # noqa: E402
from rules.ema_trend import EMATrendRule                # noqa: E402
from rules.time_of_day import TimeOfDayRule             # noqa: E402
from rules.volume_breakout import VolumeBreakoutRule    # noqa: E402
from rules.mean_reversion import MeanReversionRule      # noqa: E402
from rules.rejection_pattern import RejectionPatternRule  # noqa: E402
from utils.data_loader import load_bars                 # noqa: E402
from utils.indicators import ema, atr, rsi, bollinger_position, volume_ratio  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: build BacktestEngine from primary params
# ---------------------------------------------------------------------------
def build_primary_engine(primary_params: dict, n_contracts: int = 1) -> BacktestEngine:
    """Reconstruct the rule-based engine from config params."""
    ema_rule = EMATrendRule(
        fast_period=primary_params["ema_fast"],
        slow_period=primary_params["ema_slow"],
        min_spread_atr_ratio=primary_params.get("min_spread_atr_ratio", 0.3),
        slope_lookback=primary_params.get("slope_lookback", 3),
        atr_period=primary_params.get("atr_period", 14),
    )

    filters = [
        TimeOfDayRule(
            session_start=primary_params.get("session_start", "09:35"),
            session_end=primary_params.get("session_end", "15:45"),
            lunch_filter_enabled=False,
        ),
        VolumeBreakoutRule(
            lookback=20,
            min_ratio=primary_params.get("volume_min_ratio", 1.2),
            max_ratio=primary_params.get("volume_max_ratio", 2.0),
        ),
    ]

    confirmations = [
        MeanReversionRule(
            bb_period=primary_params.get("bb_period", 20),
            bb_std=2.0,
            long_bb_threshold=0.3,
            short_bb_threshold=0.7,
            rsi_period=primary_params.get("rsi_period", 14),
            rsi_long_threshold=35.0,
            rsi_short_threshold=65.0,
        ),
        RejectionPatternRule(min_wick_body_ratio=1.5),
    ]

    aggregator = SignalAggregator(
        primary_rule=ema_rule,
        filter_rules=filters,
        confirmation_rules=confirmations,
        min_confirmations=1,
    )

    risk_manager = RiskManager(
        contracts=n_contracts,
        point_value=5.0,
        tick_size=0.25,
        tick_value=1.25,
        max_daily_loss=-800.0,
        per_trade_max_loss=200.0 * n_contracts,
        max_consecutive_losses=3,
        cooldown_bars=5,
        flatten_minutes_before_close=5,
        drawdown_buffer=500.0,
    )

    return BacktestEngine(
        aggregator=aggregator,
        risk_manager=risk_manager,
        commission_per_side=0.62,
        slippage_ticks=1,
        profit_target_atr=primary_params["pt_atr_mult"],
        stop_loss_atr=primary_params["sl_atr_mult"],
        time_stop_bars=primary_params.get("time_stop_bars", 24),
        trailing_activation_atr=primary_params.get("trailing_activation_atr", 1.0),
        trailing_distance_atr=primary_params.get("trailing_distance_atr", 0.75),
    )


# ---------------------------------------------------------------------------
# Feature extraction for meta-labels
# ---------------------------------------------------------------------------
def build_meta_features(
    bars: pd.DataFrame,
    trade_entry_bars: List[int],
    window: int = 20,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Build feature matrix for meta-labeling secondary model.
    For each trade entry, extract features from the `window` bars prior to entry.

    Features computed on full bars series (causal), then sliced at entry_bar.
    """
    # Precompute indicators on full series
    close = bars["close"]
    high = bars["high"]
    low = bars["low"]
    volume = bars["volume"]

    atr_14 = atr(high, low, close, period=atr_period)
    rsi_14 = rsi(close, period=14)
    bb_pos = bollinger_position(close, period=20, num_std=2.0)
    vol_ratio_20 = volume_ratio(volume, lookback=20)
    ema_13 = ema(close, period=13)
    ema_34 = ema(close, period=34)

    ema_spread = (ema_13 - ema_34) / (atr_14 + 1e-9)
    log_ret_1 = close.pct_change(1)
    log_ret_5 = close.pct_change(5)
    vol_20 = close.pct_change(1).rolling(20).std()

    rows = []
    for entry_bar in trade_entry_bars:
        if entry_bar < window:
            rows.append({})
            continue

        # Lookback window: [entry_bar - window, entry_bar]
        sl = slice(entry_bar - window, entry_bar + 1)
        feat = {
            # Current values at entry
            "atr_14": float(atr_14.iloc[entry_bar]) if entry_bar < len(atr_14) else np.nan,
            "rsi_14": float(rsi_14.iloc[entry_bar]) if entry_bar < len(rsi_14) else np.nan,
            "bb_position": float(bb_pos.iloc[entry_bar]) if entry_bar < len(bb_pos) else np.nan,
            "vol_ratio": float(vol_ratio_20.iloc[entry_bar]) if entry_bar < len(vol_ratio_20) else np.nan,
            "ema_spread": float(ema_spread.iloc[entry_bar]) if entry_bar < len(ema_spread) else np.nan,

            # Recent returns
            "ret_1": float(log_ret_1.iloc[entry_bar]) if entry_bar < len(log_ret_1) else np.nan,
            "ret_5": float(log_ret_5.iloc[entry_bar]) if entry_bar < len(log_ret_5) else np.nan,

            # Volatility regime
            "vol_20": float(vol_20.iloc[entry_bar]) if entry_bar < len(vol_20) else np.nan,
            "vol_regime": float(vol_20.iloc[entry_bar] / vol_20.rolling(60).mean().iloc[entry_bar])
                         if entry_bar < len(vol_20) else np.nan,

            # Rolling stats over window
            "ret_window_mean": float(log_ret_1.iloc[sl].mean()),
            "ret_window_std": float(log_ret_1.iloc[sl].std()),
            "rsi_window_mean": float(rsi_14.iloc[sl].mean()) if entry_bar < len(rsi_14) else np.nan,
            "bb_window_mean": float(bb_pos.iloc[sl].mean()) if entry_bar < len(bb_pos) else np.nan,
            "vol_ratio_window": float(vol_ratio_20.iloc[sl].mean()) if entry_bar < len(vol_ratio_20) else np.nan,

            # Trend momentum
            "ema_spread_5ago": float(ema_spread.iloc[entry_bar - 5]) if (entry_bar - 5) >= 0 else np.nan,
            "ema_spread_trend": float(ema_spread.iloc[entry_bar] - ema_spread.iloc[max(0, entry_bar - 5)]),

            # Time features
            "hour": int(bars.index[entry_bar].hour) if entry_bar < len(bars) else np.nan,
            "minute": int(bars.index[entry_bar].minute) if entry_bar < len(bars) else np.nan,
        }
        rows.append(feat)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Secondary model training
# ---------------------------------------------------------------------------
def train_secondary_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_kind: str,
    model_params: dict,
) -> Any:
    """Train the secondary meta-labeling model."""
    # Drop rows with NaN features
    mask = X.notna().all(axis=1)
    X_clean = X[mask].copy()
    y_clean = y[mask].copy()

    if len(X_clean) < 20 or y_clean.nunique() < 2:
        logger.warning(f"Insufficient training data: n={len(X_clean)}, classes={y_clean.nunique()}")
        return None

    if model_kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**model_params)
    elif model_kind == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier(**model_params)
    elif model_kind == "lightgbm":
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(**model_params)
        except ImportError:
            logger.warning("LightGBM not available, falling back to RandomForest")
            from sklearn.ensemble import RandomForestClassifier
            params = {k: v for k, v in model_params.items()
                      if k in ["n_estimators", "max_depth", "min_samples_leaf", "random_state", "class_weight"]}
            model = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")

    model.fit(X_clean, y_clean)
    return model, mask


def evaluate_meta_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict:
    """Compute AUC and precision lift metrics for the meta model."""
    from sklearn.metrics import roc_auc_score, precision_score

    mask = X_test.notna().all(axis=1)
    X_clean = X_test[mask]
    y_clean = y_test[mask]

    if len(X_clean) == 0 or model is None:
        return {"auc": 0.5, "n_test": 0, "precision_before": 0.0, "precision_after": 0.0, "lift_pct": 0.0}

    proba = model.predict_proba(X_clean)
    if proba.shape[1] == 2:
        p_positive = proba[:, 1]
    else:
        p_positive = proba[:, 0]

    auc = roc_auc_score(y_clean, p_positive) if y_clean.nunique() > 1 else 0.5

    # Precision before meta-filter = raw win rate of primary signals
    precision_before = float(y_clean.mean())

    # Apply threshold filter
    filtered_mask = p_positive >= threshold
    if filtered_mask.sum() == 0:
        precision_after = 0.0
        n_filtered = 0
    else:
        precision_after = float(y_clean[filtered_mask].mean())
        n_filtered = int(filtered_mask.sum())

    lift_pct = (precision_after - precision_before) * 100 if precision_before > 0 else 0.0

    return {
        "auc": round(auc, 4),
        "n_test": int(len(y_clean)),
        "n_filtered": n_filtered,
        "filter_rate": round(float(filtered_mask.mean()), 4),
        "precision_before": round(precision_before, 4),
        "precision_after": round(precision_after, 4),
        "lift_pct": round(lift_pct, 2),
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------
def run_experiment(config: dict, data_path: str, output_dir: Path) -> dict:
    """Run a single batch20 meta-labeling experiment."""
    t0 = time.time()
    exp_id = config["exp_id"]
    logger.info(f"Starting {exp_id}")

    # Load data
    try:
        bars_train = load_bars(
            data_path,
            start_date=config["training_start"],
            end_date=config["training_end"],
        )
        bars_oos = load_bars(
            data_path,
            start_date=config["oos_start"],
            end_date=config["oos_end"],
        )
    except Exception as e:
        return {"exp_id": exp_id, "error": str(e), "status": "FAILED"}

    logger.info(f"Train bars: {len(bars_train):,}, OOS bars: {len(bars_oos):,}")

    primary_params = config["primary_params"]
    n_contracts = primary_params.get("n_contracts", 1)

    # ---- Step 1: Run primary signal on training data ----
    engine = build_primary_engine(primary_params, n_contracts=n_contracts)
    train_result = engine.run(bars_train, starting_equity=50_000.0)
    train_summary = train_result.summary()
    train_trades = train_result.trades

    logger.info(
        f"Primary on train: {train_summary['num_trades']} trades, "
        f"win_rate={train_summary.get('win_rate', 0):.1%}"
    )

    if len(train_trades) < config.get("min_meta_samples", 50):
        return {
            "exp_id": exp_id,
            "error": f"Too few primary trades: {len(train_trades)} < {config['min_meta_samples']}",
            "status": "INSUFFICIENT_DATA",
            "primary_train_summary": train_summary,
        }

    # ---- Step 2: Build meta features and labels on training data ----
    window = config.get("feature_window_bars", 20)
    entry_bars = [t.entry_bar for t in train_trades]
    meta_labels = [1 if t.pnl > 0 else 0 for t in train_trades]

    X_train = build_meta_features(bars_train, entry_bars, window=window)
    y_train = pd.Series(meta_labels)

    # ---- Step 3: Train secondary model ----
    secondary_kind = config["secondary_model_kind"]
    secondary_params = config["secondary_params"]
    result_fit = train_secondary_model(X_train, y_train, secondary_kind, secondary_params)

    if result_fit is None:
        return {
            "exp_id": exp_id,
            "error": "Secondary model training failed",
            "status": "TRAINING_FAILED",
        }

    model, train_mask = result_fit
    threshold = config["secondary_threshold"]

    # In-sample evaluation
    train_metrics = evaluate_meta_model(model, X_train, y_train, threshold)
    logger.info(
        f"Train meta: AUC={train_metrics['auc']:.3f}, "
        f"precision {train_metrics['precision_before']:.1%} → {train_metrics['precision_after']:.1%} "
        f"({train_metrics['lift_pct']:+.1f}% lift)"
    )

    # ---- Step 4: OOS evaluation ----
    engine_oos = build_primary_engine(primary_params, n_contracts=n_contracts)
    oos_result = engine_oos.run(bars_oos, starting_equity=50_000.0)
    oos_summary = oos_result.summary()
    oos_trades = oos_result.trades

    logger.info(f"Primary OOS: {oos_summary['num_trades']} trades, win_rate={oos_summary.get('win_rate', 0):.1%}")

    if len(oos_trades) > 5:
        oos_entry_bars = [t.entry_bar for t in oos_trades]
        oos_meta_labels = [1 if t.pnl > 0 else 0 for t in oos_trades]
        X_oos = build_meta_features(bars_oos, oos_entry_bars, window=window)
        y_oos = pd.Series(oos_meta_labels)
        oos_metrics = evaluate_meta_model(model, X_oos, y_oos, threshold)
    else:
        oos_metrics = {"error": "too few OOS trades", "auc": 0.5}

    logger.info(
        f"OOS meta: AUC={oos_metrics.get('auc', 0):.3f}, "
        f"lift={oos_metrics.get('lift_pct', 0):+.1f}%"
    )

    elapsed = time.time() - t0

    result = {
        "exp_id": exp_id,
        "status": "SUCCESS",
        "elapsed_seconds": round(elapsed, 1),
        "config": config,
        "primary": {
            "train": train_summary,
            "oos": oos_summary,
        },
        "meta": {
            "train": train_metrics,
            "oos": oos_metrics,
        },
        "success_check": {
            "meta_auc_ok": oos_metrics.get("auc", 0) >= config.get("success_thresholds", {}).get("min_meta_auc", 0.52),
            "lift_ok": oos_metrics.get("lift_pct", 0) >= config.get("success_thresholds", {}).get("min_precision_lift_pct", 5.0),
            "win_rate_after_ok": oos_metrics.get("precision_after", 0) >= config.get("success_thresholds", {}).get("min_win_rate_after_filter", 0.48),
        },
    }

    # Save result
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{exp_id}_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Result saved to {result_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run batch20 meta-labeling experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config JSON")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(_PROJECT_ROOT / "ml_intraday_v3" / "data" / "processed" / "mes_bars_databento_rth.h5"),
        help="Override data path from config",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_PROJECT_ROOT / "ml_intraday_v3" / "experiments" / "results" / "batch20"),
        help="Output directory",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    with open(args.config) as f:
        config = json.load(f)

    # Local data path overrides GCS path
    data_path = args.data_path
    if not Path(data_path).exists():
        # Try default relative paths
        for alt in [
            _PROJECT_ROOT / "ml_intraday_v3" / "data" / "processed" / "mes_bars_databento_rth.h5",
            _PROJECT_ROOT / "data" / "processed" / "mes_bars_databento_rth.h5",
        ]:
            if alt.exists():
                data_path = str(alt)
                break

    result = run_experiment(config, data_path=data_path, output_dir=Path(args.output))

    # Print summary
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: {result['exp_id']}")
    print("=" * 60)
    if result.get("status") == "SUCCESS":
        meta_oos = result.get("meta", {}).get("oos", {})
        primary_oos = result.get("primary", {}).get("oos", {})
        checks = result.get("success_check", {})
        print(f"Status:     {result['status']}")
        print(f"Primary OOS trades: {primary_oos.get('num_trades', 0)}")
        print(f"Primary OOS win rate: {primary_oos.get('win_rate', 0):.1%}")
        print(f"Meta AUC (OOS):  {meta_oos.get('auc', 0):.3f} {'✓' if checks.get('meta_auc_ok') else '✗'}")
        print(f"Precision lift:  {meta_oos.get('lift_pct', 0):+.1f}% {'✓' if checks.get('lift_ok') else '✗'}")
        print(f"Win rate after:  {meta_oos.get('precision_after', 0):.1%} {'✓' if checks.get('win_rate_after_ok') else '✗'}")
        all_pass = all(checks.values()) if checks else False
        print(f"\nAll criteria met: {'YES - QUALIFIED' if all_pass else 'NO'}")
    else:
        print(f"Status:     {result.get('status', 'UNKNOWN')}")
        print(f"Error:      {result.get('error', 'unknown')}")


if __name__ == "__main__":
    main()
