"""
Combine Oracle — Discovers market inefficiencies through a dynamic research loop.

Workflow:
  1. Load microstructure + OHLCV data
  2. Run information-theoretic feature screening (mutual information vs forward returns)
  3. Walk-forward model training with combine-aware objective
  4. Monte Carlo combine simulation per fold
  5. Adaptive threshold search: find the probability cutoff that maximizes P(pass)
  6. Regime-conditional analysis: identify when the model has vs lacks edge
  7. Output: ranked feature candidates, optimal threshold, P(pass) curve

This is the research loop — run it whenever new data arrives or after a combine attempt.

Usage:
  python ml_intraday_v3/scripts/combine_oracle.py [--quick]

  --quick: Skip walk-forward, use simple train/test split (faster)
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE  = Path(__file__).parents[2]
DATA  = BASE / "data" / "processed"
RES   = BASE / "ml_intraday_v3" / "results"
RES.mkdir(exist_ok=True)

try:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.isotonic import IsotonicRegression
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

# ── Combine config ─────────────────────────────────────────────────────────
COMBINE = dict(
    profit_target     = 3000,
    daily_loss_limit  = 1000,
    trailing_dd_limit = 2000,
    min_trading_days  = 5,
    max_days          = 30,
)

# ── All candidate features grouped by type ─────────────────────────────────
FEATURE_GROUPS = {
    "institutional_flow": [
        "lg_ofi_imb", "lg_sm_diverge", "large_frac", "lg_ofi",
    ],
    "flow_dynamics": [
        "ofi_imb", "ofi_accel", "ofi_early", "ofi_late",
    ],
    "price_impact": [
        "kyles_lambda", "roll_spread",
    ],
    "trade_activity": [
        "trade_rate", "max_run", "avg_size", "max_size", "size_std",
    ],
    "ohlcv_momentum": [
        "ret_1", "ret_3", "ret_6", "rsi_14", "ema9_ratio", "ema21_ratio",
    ],
    "ohlcv_structure": [
        "norm_range", "norm_body", "vol_z", "atr_z", "vwap_dev",
    ],
    "time": [
        "hour_sin", "hour_cos", "dow",
    ],
}
ALL_CANDIDATES = [f for g in FEATURE_GROUPS.values() for f in g]


def compute_atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()


def rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p - 1, min_periods=p).mean()
    l = (-d.clip(upper=0)).ewm(com=p - 1, min_periods=p).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add OHLCV-derived features and normalize raw microstructure features."""
    df = df.copy()
    c   = df["close"]
    atr = compute_atr(df)
    df["atr"] = atr
    window = 78

    # OHLCV
    for n in [1, 3, 6]:
        df[f"ret_{n}"] = np.log(c / c.shift(n))
    df["rsi_14"] = rsi(c, 14)
    ema9  = c.ewm(span=9,  min_periods=9).mean()
    ema21 = c.ewm(span=21, min_periods=21).mean()
    df["ema9_ratio"]  = (c / ema9  - 1) * 100
    df["ema21_ratio"] = (c / ema21 - 1) * 100
    df["norm_range"]  = (df["high"] - df["low"]) / atr.replace(0, np.nan)
    df["norm_body"]   = (c - df["open"]) / atr.replace(0, np.nan)

    tv = df.get("total_vol", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    df["vol_z"] = (tv - tv.rolling(window).mean()) / tv.rolling(window).std().replace(0, np.nan)
    df["atr_z"] = (atr - atr.rolling(window).mean()) / atr.rolling(window).std().replace(0, np.nan)

    if "vwap" in df.columns:
        df["vwap_dev"] = (c - df["vwap"]) / atr.replace(0, np.nan)
    else:
        df["vwap_dev"] = 0.0

    h_et = (df.index.hour - 5) % 24
    df["hour_sin"] = np.sin(2 * np.pi * h_et / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h_et / 24)
    df["dow"]      = df.index.dayofweek.astype(float)

    # Normalize microstructure features to rolling z-scores
    flow_cols = ["ofi_imb", "lg_ofi_imb", "ofi_accel", "ofi_early", "ofi_late",
                 "kyles_lambda", "roll_spread", "trade_rate", "avg_size", "max_size", "size_std"]
    for col in flow_cols:
        if col in df.columns:
            ma = df[col].rolling(window, min_periods=20).mean()
            sd = df[col].rolling(window, min_periods=20).std().replace(0, np.nan)
            df[col] = (df[col] - ma) / sd

    return df


def label_triple_barrier(df: pd.DataFrame, pt_atr: float, sl_atr: float, hor: int) -> pd.Series:
    c   = df["close"].values
    hi  = df["high"].values
    lo  = df["low"].values
    atr = df["atr"].values
    n   = len(df)
    ll  = np.full(n, -1, dtype=np.int8)
    for i in range(n - hor):
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue
        ll[i] = 0
        pt = c[i] + pt_atr * a
        sl = c[i] - sl_atr * a
        for j in range(i + 1, i + hor + 1):
            if hi[j] >= pt:
                ll[i] = 1; break
            elif lo[j] <= sl:
                break
    return pd.Series(ll, index=df.index, name="label")


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Information-theoretic feature screening
# ═══════════════════════════════════════════════════════════════════════════

def screen_features(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """
    Rank all candidate features by mutual information with forward labels.
    MI is non-parametric — captures nonlinear relationships OHLCV-only models miss.
    """
    available = [f for f in ALL_CANDIDATES if f in df.columns]
    valid     = labels >= 0

    X = df.loc[valid, available].fillna(0).astype(np.float32)
    y = labels[valid].astype(int)

    print(f"    Computing MI for {len(available)} features on {len(X)} samples...")
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)

    rows = []
    for feat, score in zip(available, mi):
        group = next((g for g, fs in FEATURE_GROUPS.items() if feat in fs), "unknown")
        rows.append({"feature": feat, "mi_score": round(float(score), 6), "group": group})

    result = pd.DataFrame(rows).sort_values("mi_score", ascending=False).reset_index(drop=True)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Walk-forward training + evaluation
# ═══════════════════════════════════════════════════════════════════════════

def walk_forward_train(
    df: pd.DataFrame,
    labels: pd.Series,
    features: list,
    n_folds: int = 3,
) -> tuple:
    """
    Purged walk-forward: train on first K months, test on next M months.
    Returns list of (fold_name, oos_probs, oos_labels, oos_df) tuples.
    """
    dates = df.index.normalize().unique().sort_values()
    n_days = len(dates)
    fold_size = n_days // (n_folds + 1)
    embargo_days = 5  # purge 5 days around train/test boundary to prevent leakage

    folds = []
    for fold in range(n_folds):
        train_end_idx = (fold + 1) * fold_size
        test_start_idx = train_end_idx + embargo_days
        test_end_idx   = test_start_idx + fold_size

        if test_end_idx > n_days:
            break

        train_end   = dates[train_end_idx - 1]
        test_start  = dates[min(test_start_idx, n_days - 1)]
        test_end    = dates[min(test_end_idx - 1, n_days - 1)]

        train_mask = (df.index <= pd.Timestamp(str(train_end), tz="UTC"))
        test_mask  = (
            (df.index >= pd.Timestamp(str(test_start), tz="UTC")) &
            (df.index <= pd.Timestamp(str(test_end),   tz="UTC"))
        )

        label_tr = labels[train_mask & (labels >= 0)]
        label_te = labels[test_mask  & (labels >= 0)]
        if len(label_tr) < 100 or len(label_te) < 20:
            continue

        feat_ok_tr = df.loc[label_tr.index, features].notna().all(axis=1)
        feat_ok_te = df.loc[label_te.index, features].notna().all(axis=1)
        X_tr = df.loc[label_tr.index[feat_ok_tr], features].astype(np.float32)
        y_tr = label_tr[feat_ok_tr].astype(int)
        X_te = df.loc[label_te.index[feat_ok_te], features].astype(np.float32)
        y_te = label_te[feat_ok_te].astype(int)

        split = int(len(X_tr) * 0.85)
        model = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=40,
            min_child_samples=30, subsample=0.75, colsample_bytree=0.75,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        )
        model.fit(
            X_tr.iloc[:split], y_tr.iloc[:split],
            eval_set=[(X_tr.iloc[split:], y_tr.iloc[split:])],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)],
        )

        raw_probs = model.predict_proba(X_te)[:, 1]
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(model.predict_proba(X_tr.iloc[split:])[:, 1], y_tr.iloc[split:])
        cal_probs = cal.predict(raw_probs)

        fold_auc = roc_auc_score(y_te, cal_probs) if len(y_te.unique()) > 1 else 0.5
        fold_name = f"fold_{fold+1}_{str(test_start)[:10]}_{str(test_end)[:10]}"
        print(f"    {fold_name}: AUC={fold_auc:.4f}, n={len(y_te)}")

        folds.append((fold_name, cal_probs, y_te.values, df.loc[X_te.index].copy(), model))

    return folds


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Threshold search — maximize P(pass combine)
# ═══════════════════════════════════════════════════════════════════════════

def threshold_search(
    df: pd.DataFrame,
    probs: np.ndarray,
    n_mc: int = 5000,
) -> dict:
    """
    Grid search over probability thresholds to find the one that maximizes P(pass).
    This replaces the ad-hoc 0.55 default with a combine-optimal cutoff.
    """
    df = df.copy()
    df["p_long"] = probs

    best = {"threshold": 0.55, "p_pass": 0.0}
    results = []

    for thresh in np.arange(0.50, 0.85, 0.025):
        signals = df[df["p_long"] >= thresh].copy()
        if len(signals) < 5:
            continue

        # Simulate trades at this threshold
        trades = _simulate_at_threshold(df, probs, thresh)
        if not trades:
            continue

        mc = _fast_mc(trades, n_mc)
        results.append({
            "threshold": round(float(thresh), 3),
            "n_trades_oos": len(trades),
            "p_pass": mc["p_pass"],
            "p95_dd": mc["p95_dd"],
            "median_days": mc["median_days"],
        })
        if mc["p_pass"] > best["p_pass"] and mc["p95_dd"] <= COMBINE["trailing_dd_limit"]:
            best = {"threshold": round(float(thresh), 3), **mc}

    return {"best": best, "grid": results}


def _simulate_at_threshold(df: pd.DataFrame, probs: np.ndarray, thresh: float) -> list:
    """Lightweight backtest at a given threshold."""
    df = df.copy()
    df["p_long"] = probs
    pt = 2.5; sl = 1.25; hor = 8
    c = df["close"].values; h = df["high"].values
    l = df["low"].values; a = df["atr"].values
    p = df["p_long"].values; idx = df.index

    trades = []; in_t = False; cool = 0; ep = ea = 0.0; eb = 0
    daily_pnl: dict = {}
    for i in range(len(df)):
        ds = str(idx[i].date())
        daily_pnl.setdefault(ds, 0.0)
        if in_t:
            ptp = ep + pt * ea; slp = ep - sl * ea
            if h[i] >= ptp:       pts, why = pt * ea, "PT"
            elif l[i] <= slp:     pts, why = -sl * ea, "SL"
            elif (i - eb) >= hor: pts, why = c[i] - ep, "TIME"
            else: continue
            dol = pts * 2.0 * 2  # 2 contracts × $2/pt
            daily_pnl[ds] += dol
            trades.append({"pnl_dollars": dol, "entry_time": str(idx[eb]), "date": ds})
            in_t = False; cool = 2; continue
        if cool > 0: cool -= 1; continue
        if daily_pnl[ds] <= -800: continue
        if np.isnan(a[i]) or a[i] < 4: continue
        h_et = (idx[i].hour - 5) % 24
        if h_et < 10: continue
        if idx[i].weekday() == 3: continue
        if p[i] >= thresh:
            in_t = True; ep = c[i]; ea = a[i]; eb = i
    return trades


def _fast_mc(trades: list, n: int = 5000) -> dict:
    rng = np.random.default_rng(42)
    tdf = pd.DataFrame(trades)
    daily = tdf.groupby("date")["pnl_dollars"].sum().values
    target = COMBINE["profit_target"]
    dd_lim = COMBINE["trailing_dd_limit"]
    day_lim= COMBINE["daily_loss_limit"]
    n_pass = 0; days_to_pass = []; max_dds = []
    for _ in range(n):
        pnl = 0.0; peak = 0.0; days = 0; passed = False
        for d in rng.choice(daily, size=COMBINE["max_days"], replace=True):
            d = max(d, -day_lim)
            pnl += d; peak = max(peak, pnl)
            dd = pnl - peak; days += 1
            if dd <= -dd_lim or pnl <= -dd_lim: break
            if pnl >= target and days >= COMBINE["min_trading_days"]:
                passed = True; days_to_pass.append(days); break
        max_dds.append(abs(pnl - peak))
        if passed: n_pass += 1
    return {
        "p_pass":      round(n_pass / n, 4),
        "p95_dd":      round(float(np.percentile(max_dds, 95)), 2),
        "median_days": int(np.median(days_to_pass)) if days_to_pass else COMBINE["max_days"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Regime-conditional edge analysis
# ═══════════════════════════════════════════════════════════════════════════

def regime_edge_analysis(df: pd.DataFrame, probs: np.ndarray, labels: np.ndarray) -> dict:
    """
    Identify WHEN the model has edge vs. when it's noise.
    Splits the OOS period by: time of day, day of week, volatility regime, OFI regime.
    Report AUC per regime — only trade regimes where AUC > 0.52.
    """
    df = df.copy()
    df["p_long"] = probs
    df["label"]  = labels

    valid = df["label"] >= 0
    df = df[valid]

    results = {}

    # By hour (ET)
    h_et = (df.index.hour - 5) % 24
    for h in range(10, 16):
        mask = (h_et == h)
        if mask.sum() < 20:
            continue
        auc = roc_auc_score(df.loc[mask, "label"], df.loc[mask, "p_long"]) if df.loc[mask, "label"].nunique() > 1 else 0.5
        results[f"hour_{h:02d}ET"] = {"n": int(mask.sum()), "auc": round(auc, 4)}

    # By day of week
    for dow, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"]):
        mask = df.index.dayofweek == dow
        if mask.sum() < 20:
            continue
        auc = roc_auc_score(df.loc[mask, "label"], df.loc[mask, "p_long"]) if df.loc[mask, "label"].nunique() > 1 else 0.5
        results[f"dow_{name}"] = {"n": int(mask.sum()), "auc": round(auc, 4)}

    # By ATR regime (high vs low vol)
    if "atr_z" in df.columns:
        high_vol = df["atr_z"] > 0.5
        low_vol  = df["atr_z"] < -0.5
        for name, mask in [("high_vol", high_vol), ("low_vol", low_vol)]:
            if mask.sum() < 20:
                continue
            auc = roc_auc_score(df.loc[mask, "label"], df.loc[mask, "p_long"]) if df.loc[mask, "label"].nunique() > 1 else 0.5
            results[f"vol_{name}"] = {"n": int(mask.sum()), "auc": round(auc, 4)}

    # By OFI regime (institutional flow dominant vs not)
    if "lg_ofi_imb" in df.columns:
        ofi_clean = df["lg_ofi_imb"].fillna(0)
        pos_flow = ofi_clean > 0.5
        neg_flow = ofi_clean < -0.5
        for name, mask in [("bullish_inst_flow", pos_flow), ("bearish_inst_flow", neg_flow)]:
            if mask.sum() < 20:
                continue
            auc = roc_auc_score(df.loc[mask, "label"], df.loc[mask, "p_long"]) if df.loc[mask, "label"].nunique() > 1 else 0.5
            results[f"ofi_{name}"] = {"n": int(mask.sum()), "auc": round(auc, 4)}

    return dict(sorted(results.items(), key=lambda x: -x[1]["auc"]))


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Simple train/test split (faster)")
    args = parser.parse_args()

    if not HAS_DEPS:
        print("Install: pip install lightgbm scikit-learn"); return

    print("=" * 70)
    print("  Combine Oracle — Dynamic Market Inefficiency Discovery Loop")
    print("=" * 70)

    micro_path = DATA / "mnq_microstructure_5min.parquet"
    if not micro_path.exists():
        print(f"ERROR: {micro_path} not found. Run build_microstructure_features.py first.")
        return

    print("\n[1] Loading and featurizing data...")
    df = pd.read_parquet(micro_path)
    df.index = pd.to_datetime(df.index, utc=True)
    df = build_all_features(df)
    print(f"    {len(df)} bars  |  {df.index[0].date()} → {df.index[-1].date()}")

    print("\n[2] Labeling (PT=2.5×ATR, SL=1.25×ATR, horizon=8 bars)...")
    labels = label_triple_barrier(df, pt_atr=2.5, sl_atr=1.25, hor=8)
    valid = labels >= 0
    print(f"    {valid.sum()} labeled bars, pos_rate={labels[valid].mean():.3f}")

    print("\n[3] Information-theoretic feature screening...")
    mi_df = screen_features(df[valid], labels[valid])
    print(f"\n    Top features by Mutual Information:")
    for _, row in mi_df.head(15).iterrows():
        tag = "(MICRO)" if row["group"] not in ("ohlcv_momentum", "ohlcv_structure", "time") else "      "
        print(f"    {tag}  [{row['group']:22s}]  {row['feature']:25s}: MI={row['mi_score']:.6f}")

    # Select top features by MI (keep those with MI > threshold)
    mi_threshold = max(mi_df["mi_score"].quantile(0.5), 1e-5)
    selected     = mi_df[mi_df["mi_score"] >= mi_threshold]["feature"].tolist()
    available    = [f for f in selected if f in df.columns]
    print(f"\n    Selected {len(available)} features above MI threshold {mi_threshold:.6f}")

    if args.quick:
        print("\n[4] Quick train/test split (--quick mode)...")
        t_cut = pd.Timestamp("2026-01-31 23:59", tz="UTC")
        o_start = pd.Timestamp("2026-02-01", tz="UTC")
        train_lab = labels[df.index <= t_cut]
        oos_lab   = labels[df.index >= o_start]
        train_df  = df[df.index <= t_cut]
        oos_df    = df[df.index >= o_start]

        # Quick fit
        feat_ok = train_df.loc[train_lab[train_lab >= 0].index, available].notna().all(axis=1)
        X_tr = train_df.loc[train_lab[train_lab >= 0].index[feat_ok], available].astype(np.float32)
        y_tr = train_lab[train_lab >= 0][feat_ok].astype(int)

        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=40,
            min_child_samples=30, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        )
        split = int(len(X_tr) * 0.85)
        model.fit(
            X_tr.iloc[:split], y_tr.iloc[:split],
            eval_set=[(X_tr.iloc[split:], y_tr.iloc[split:])],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)],
        )

        oos_feat_ok = oos_df.loc[oos_lab[oos_lab >= 0].index, available].notna().all(axis=1)
        X_oos_eval = oos_df.loc[oos_lab[oos_lab >= 0].index[oos_feat_ok], available].astype(np.float32)
        y_oos_eval = oos_lab[oos_lab >= 0][oos_feat_ok].astype(int)

        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(model.predict_proba(X_tr.iloc[split:])[:, 1], y_tr.iloc[split:])

        raw_oos = model.predict_proba(X_oos_eval)[:, 1]
        cal_oos = cal.predict(raw_oos)
        oos_auc = roc_auc_score(y_oos_eval, cal_oos) if y_oos_eval.nunique() > 1 else 0.5
        print(f"    OOS AUC: {oos_auc:.4f}")

        fold_probs  = cal_oos
        fold_labels = y_oos_eval.values
        fold_df     = oos_df.loc[X_oos_eval.index].copy()
        fold_df["atr"] = fold_df.get("atr", compute_atr(fold_df))

    else:
        print("\n[4] Purged walk-forward training...")
        folds = walk_forward_train(df, labels, available, n_folds=3)
        if not folds:
            print("    ERROR: no folds completed (insufficient data)")
            return

        # Combine all fold results for threshold search
        fold_probs  = np.concatenate([f[1] for f in folds])
        fold_labels = np.concatenate([f[2] for f in folds])
        fold_df     = pd.concat([f[3] for f in folds])
        fold_df["atr"] = compute_atr(fold_df)
        fold_auc = roc_auc_score(fold_labels, fold_probs) if len(np.unique(fold_labels)) > 1 else 0.5
        print(f"\n    Combined fold AUC: {fold_auc:.4f}")

    print("\n[5] Threshold grid search (optimizing P(pass combine))...")
    thresh_result = threshold_search(fold_df, fold_probs)
    best_t = thresh_result["best"]
    print(f"\n    Optimal threshold: {best_t['threshold']}")
    print(f"    P(pass) at optimal: {best_t['p_pass']*100:.1f}%")
    print(f"    p95 max DD:         ${best_t.get('p95_dd', 0):,.0f}")
    print(f"    Median days:        {best_t.get('median_days', 0)}")

    print("\n    Threshold grid:")
    for row in thresh_result["grid"]:
        marker = " ← OPTIMAL" if row["threshold"] == best_t["threshold"] else ""
        print(f"    thresh={row['threshold']:.3f}: P(pass)={row['p_pass']*100:.1f}%  "
              f"p95_dd=${row['p95_dd']:,.0f}  n={row['n_trades_oos']}{marker}")

    print("\n[6] Regime-conditional edge analysis...")
    regime_results = regime_edge_analysis(fold_df, fold_probs, fold_labels)
    print("\n    AUC by regime (top = most edge-rich):")
    for regime, stats in list(regime_results.items())[:12]:
        edge_tag = " ← TRADE THIS" if stats["auc"] > 0.52 else (" ← AVOID" if stats["auc"] < 0.49 else "")
        print(f"    {regime:35s}: AUC={stats['auc']:.4f}  n={stats['n']}{edge_tag}")

    # Identify edge-positive regimes
    good_regimes  = {k: v for k, v in regime_results.items() if v["auc"] > 0.52}
    avoid_regimes = {k: v for k, v in regime_results.items() if v["auc"] < 0.49}
    print(f"\n    Edge-positive regimes ({len(good_regimes)}): {list(good_regimes.keys())}")
    print(f"    Avoid regimes ({len(avoid_regimes)}):         {list(avoid_regimes.keys())}")

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "mi_ranking":       mi_df.to_dict(orient="records"),
        "selected_features": available,
        "micro_feature_count": len([f for f in available
                                    if f in [x for g in ["institutional_flow", "flow_dynamics",
                                                         "price_impact", "trade_activity"]
                                             for x in FEATURE_GROUPS[g]]]),
        "threshold_search":  thresh_result,
        "regime_analysis":   regime_results,
        "edge_positive_regimes": good_regimes,
        "avoid_regimes":         avoid_regimes,
        "combine_config":        COMBINE,
    }

    out_path = RES / "combine_oracle_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    print(f"\n{'='*70}")
    print("  ORACLE VERDICT")
    print(f"{'='*70}")
    print(f"  Best features (by MI):  {mi_df.head(5)['feature'].tolist()}")
    print(f"  Optimal threshold:      {best_t['threshold']}")
    print(f"  P(pass combine):        {best_t['p_pass']*100:.1f}%")
    print(f"  Next step: run informed_flow_ml.py with threshold={best_t['threshold']}")
    if good_regimes:
        print(f"  High-edge regimes:      {list(good_regimes.keys())[:3]}")
    if avoid_regimes:
        print(f"  Avoid regimes:          {list(avoid_regimes.keys())[:3]}")


if __name__ == "__main__":
    main()
