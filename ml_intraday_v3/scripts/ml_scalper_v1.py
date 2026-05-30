"""
Pure ML Intraday Scalper v1 — MNQ 5-min bars
==============================================
Train: Aug–Nov 2025 (5-min ETH bars, OFI where available)
Val  : Dec 2025
OOS  : Jan–Mar 2026 (1-min RTH resampled to 5-min)

Target: high-frequency LightGBM directional model
Risk  : max daily drawdown $4,000, position sizing N contracts
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).parents[2]
DATA = BASE / "data" / "processed"
OFI_DIR = DATA / "ofi_1min"
RESULTS_DIR = BASE / "ml_intraday_v3" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────
CFG = dict(
    # Bars
    bar_freq="5min",
    rth_start=9,   # 9:30 ET = 14:30 UTC
    rth_end=16,    # 16:00 ET

    # Triple-barrier
    pt_atr=2.0,
    sl_atr=1.0,
    horizon_bars=12,   # 60 min look-ahead
    atr_period=14,

    # Model
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,

    # Purge / embargo
    purge_bars=12,   # embargo between train/test

    # Signal
    long_threshold=0.54,
    short_threshold=0.54,
    min_atr_filter=3.0,   # skip low-vol bars (ATR < this many pts)

    # Risk per trade
    n_contracts=10,       # MNQ contracts
    pt_mult=2.0,          # PT in ATR units (matches label)
    sl_mult=1.0,
    max_trades_per_day=6,
    max_daily_loss_pts=200,  # 200 pts × $2 × 10c = $4,000 daily limit
    cooldown_bars=3,
)

MNQ_PNL_PER_POINT = 2.0   # $ per MNQ point per contract


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_bars_5min() -> pd.DataFrame:
    """Load 5-min ETH bars (Aug 2025 - Apr 2026)."""
    path = DATA / "mnq_aug2025_apr2026_1min_eth.h5"
    with pd.HDFStore(str(path), "r") as s:
        key = s.keys()[0]
        df = s[key]
    df.index = pd.to_datetime(df.index, utc=True)
    print(f"  5-min ETH bars: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    return df


def load_oos_1min_rth() -> pd.DataFrame:
    """Load 1-min RTH bars (Jan-Mar 2026), resample to 5-min."""
    path = DATA / "mnq_2026ytd_1min.h5"
    with pd.HDFStore(str(path), "r") as s:
        key = s.keys()[0]
        df = s[key]
    df.index = pd.to_datetime(df.index, utc=True)
    # Resample to 5-min
    df5 = df.resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()
    print(f"  OOS 5-min bars (resampled): {len(df5)} rows, {df5.index[0].date()} to {df5.index[-1].date()}")
    return df5


def load_ofi() -> pd.DataFrame:
    """Concatenate all OFI parquets, resample to 5-min."""
    parts = []
    for fname in sorted(OFI_DIR.glob("*.parquet")):
        df = pd.read_parquet(fname)
        df.index = pd.to_datetime(df.index, utc=True)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    ofi = pd.concat(parts).sort_index()
    ofi = ofi[~ofi.index.duplicated(keep="last")]
    # Resample 1-min OFI to 5-min
    ofi5 = ofi.resample("5min").agg({
        "buy_vol": "sum", "sell_vol": "sum",
        "ofi": "sum", "total_vol": "sum", "n_trades": "sum",
    })
    ofi5["ofi_imb"] = ofi5["ofi"] / ofi5["total_vol"].replace(0, np.nan)
    print(f"  OFI 5-min: {len(ofi5)} rows, {ofi5.index[0].date()} to {ofi5.index[-1].date()}")
    return ofi5


# ── Feature Engineering ──────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features(bars: pd.DataFrame, ofi: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    close = df["close"]
    atr = compute_atr(df, CFG["atr_period"])
    df["atr"] = atr

    # Returns
    for n in [1, 3, 5, 12, 24]:
        df[f"ret_{n}"] = np.log(close / close.shift(n))

    # RSI
    df["rsi_5"] = rsi(close, 5)
    df["rsi_14"] = rsi(close, 14)

    # EMA ratios
    ema9 = close.ewm(span=9, min_periods=9).mean()
    ema21 = close.ewm(span=21, min_periods=21).mean()
    ema50 = close.ewm(span=50, min_periods=50).mean()
    df["ema9_ratio"] = (close / ema9 - 1) * 100
    df["ema21_ratio"] = (close / ema21 - 1) * 100
    df["ema9_21_cross"] = (ema9 / ema21 - 1) * 100
    df["ema21_50_cross"] = (ema21 / ema50 - 1) * 100

    # Normalized range + body
    df["norm_range"] = (df["high"] - df["low"]) / atr
    df["norm_body"] = (close - df["open"]) / atr.replace(0, np.nan)
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / atr.replace(0, np.nan)
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / atr.replace(0, np.nan)

    # Volume
    vol_ma = df["volume"].rolling(20).mean()
    df["vol_zscore"] = (df["volume"] - vol_ma) / df["volume"].rolling(20).std()

    # Rolling high/low breakout (20-bar range position)
    high20 = df["high"].rolling(20).max()
    low20 = df["low"].rolling(20).min()
    df["range_pos"] = (close - low20) / (high20 - low20).replace(0, np.nan)

    # Time features
    hour_et = (df.index.hour - 5) % 24   # rough ET offset from UTC (ignores DST)
    df["hour_sin"] = np.sin(2 * np.pi * hour_et / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour_et / 24)
    df["dow"] = df.index.dayofweek.astype(float)
    df["is_first_hour"] = ((hour_et >= 9) & (hour_et < 10)).astype(float)
    df["is_last_hour"] = ((hour_et >= 15) & (hour_et < 16)).astype(float)

    # OFI features
    if not ofi.empty:
        df = df.join(ofi[["ofi_imb", "ofi", "total_vol"]], how="left")
        df["ofi_imb"] = df["ofi_imb"].fillna(0)
        df["ofi_5"] = df["ofi"].rolling(5).sum().fillna(0)
        df["ofi_norm"] = df["ofi_5"] / df["total_vol"].rolling(5).sum().replace(0, np.nan).fillna(0)
        df.drop(columns=["ofi", "total_vol"], inplace=True)
    else:
        df["ofi_imb"] = 0.0
        df["ofi_5"] = 0.0
        df["ofi_norm"] = 0.0

    return df


# ── Triple-Barrier Labeling ──────────────────────────────────────────────────

def label_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each bar compute:
      long_label:  1 if LONG PT hit before SL within horizon, else 0
      short_label: 1 if SHORT PT hit before SL within horizon, else 0
      meta_label:  1 if either direction wins (for meta-model / filtering)
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr"].values
    n = len(df)
    horizon = CFG["horizon_bars"]
    pt_mult = CFG["pt_atr"]
    sl_mult = CFG["sl_atr"]

    long_labels = np.full(n, -1, dtype=np.int8)
    short_labels = np.full(n, -1, dtype=np.int8)
    long_ret = np.full(n, np.nan)
    short_ret = np.full(n, np.nan)

    for i in range(n - horizon):
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue
        entry = close[i]
        long_pt = entry + pt_mult * a
        long_sl = entry - sl_mult * a
        short_pt = entry - pt_mult * a
        short_sl = entry + sl_mult * a

        ll, sl_val, lr = 0, 0, 0.0
        ss, ssl, sr = 0, 0, 0.0

        for j in range(i + 1, min(i + horizon + 1, n)):
            h, l_ = high[j], low[j]
            if ll == 0:
                if h >= long_pt:
                    ll = 1; lr = pt_mult * a / entry
                    break
                elif l_ <= long_sl:
                    ll = 0; lr = -sl_mult * a / entry
                    break
            if ss == 0:
                if l_ <= short_pt:
                    ss = 1; sr = pt_mult * a / entry
                    break
                elif h >= short_sl:
                    ss = 0; sr = -sl_mult * a / entry
                    break

        long_labels[i] = ll
        short_labels[i] = ss
        long_ret[i] = lr
        short_ret[i] = sr

    # Re-run cleanly (above loop broke early — do proper separate loops)
    long_labels = np.full(n, 0, dtype=np.int8)
    short_labels = np.full(n, 0, dtype=np.int8)
    long_ret = np.zeros(n)
    short_ret = np.zeros(n)

    for i in range(n - horizon):
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue
        entry = close[i]
        long_pt = entry + pt_mult * a
        long_sl = entry - sl_mult * a
        short_pt = entry - pt_mult * a
        short_sl = entry + sl_mult * a

        for j in range(i + 1, min(i + horizon + 1, n)):
            h, l_ = high[j], low[j]
            if h >= long_pt:
                long_labels[i] = 1
                long_ret[i] = pt_mult * a
                break
            elif l_ <= long_sl:
                long_labels[i] = 0
                long_ret[i] = -sl_mult * a
                break

        for j in range(i + 1, min(i + horizon + 1, n)):
            h, l_ = high[j], low[j]
            if l_ <= short_pt:
                short_labels[i] = 1
                short_ret[i] = pt_mult * a
                break
            elif h >= short_sl:
                short_labels[i] = 0
                short_ret[i] = -sl_mult * a
                break

    df = df.copy()
    df["long_label"] = long_labels
    df["short_label"] = short_labels
    df["long_ret_pts"] = long_ret
    df["short_ret_pts"] = short_ret
    # Only valid where ATR and horizon exist
    valid = df["atr"].notna() & (df["atr"] > 0)
    df.loc[~valid, ["long_label", "short_label"]] = -1
    return df


# ── Model Training ───────────────────────────────────────────────────────────

FEATURE_COLS = [
    "ret_1", "ret_3", "ret_5", "ret_12", "ret_24",
    "rsi_5", "rsi_14",
    "ema9_ratio", "ema21_ratio", "ema9_21_cross", "ema21_50_cross",
    "norm_range", "norm_body", "upper_wick", "lower_wick",
    "vol_zscore", "range_pos",
    "hour_sin", "hour_cos", "dow", "is_first_hour", "is_last_hour",
    "ofi_imb", "ofi_5", "ofi_norm",
]


def train_model(train_df: pd.DataFrame, direction: str = "long"):
    try:
        import lightgbm as lgb
    except ImportError:
        print("lightgbm not installed — run: pip install lightgbm")
        sys.exit(1)

    label_col = f"{direction}_label"
    mask = (train_df[label_col] >= 0) & train_df[FEATURE_COLS].notna().all(axis=1)
    X = train_df.loc[mask, FEATURE_COLS].astype(np.float32)
    y = train_df.loc[mask, label_col].astype(np.int32)

    print(f"  [{direction}] train samples: {len(X)}, positive rate: {y.mean():.3f}")

    # Time-series split: last 20% as internal val
    split = int(len(X) * 0.8)
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y.iloc[:split], y.iloc[split:]

    model = lgb.LGBMClassifier(
        n_estimators=CFG["n_estimators"],
        learning_rate=CFG["learning_rate"],
        num_leaves=CFG["num_leaves"],
        min_child_samples=CFG["min_child_samples"],
        subsample=CFG["subsample"],
        colsample_bytree=CFG["colsample_bytree"],
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    from sklearn.metrics import roc_auc_score
    val_preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)
    print(f"  [{direction}] val AUC: {auc:.4f}, best_iter: {model.best_iteration_}")
    return model


# ── OOS Backtest ─────────────────────────────────────────────────────────────

def backtest(oos_df: pd.DataFrame, long_model, short_model) -> pd.DataFrame:
    """Event-driven bar-by-bar backtest with risk management."""
    feat_valid = oos_df[FEATURE_COLS].notna().all(axis=1)
    oos_df = oos_df[feat_valid].copy()

    X = oos_df[FEATURE_COLS].astype(np.float32)
    oos_df["p_long"] = long_model.predict_proba(X)[:, 1]
    oos_df["p_short"] = short_model.predict_proba(X)[:, 1]

    trades = []
    daily_pnl: dict[str, float] = {}
    daily_trades: dict[str, int] = {}
    in_trade = False
    cooldown = 0
    entry_price = 0.0
    entry_dir = 0
    entry_atr = 0.0
    entry_bar = 0

    close = oos_df["close"].values
    high = oos_df["high"].values
    low = oos_df["low"].values
    atr = oos_df["atr"].values
    p_long = oos_df["p_long"].values
    p_short = oos_df["p_short"].values
    idx = oos_df.index

    n_contracts = CFG["n_contracts"]
    pt_mult = CFG["pt_mult"]
    sl_mult = CFG["sl_mult"]
    long_th = CFG["long_threshold"]
    short_th = CFG["short_threshold"]
    max_dd = CFG["max_daily_loss_pts"]
    max_td = CFG["max_trades_per_day"]
    min_atr = CFG["min_atr_filter"]
    horizon = CFG["horizon_bars"]

    for i in range(len(oos_df)):
        date_str = str(idx[i].date())
        daily_pnl.setdefault(date_str, 0.0)
        daily_trades.setdefault(date_str, 0)

        # Check exit if in trade
        if in_trade:
            a = entry_atr
            if entry_dir == 1:
                pt_price = entry_price + pt_mult * a
                sl_price = entry_price - sl_mult * a
                if high[i] >= pt_price:
                    pnl_pts = pt_mult * a
                    hit = "PT"
                elif low[i] <= sl_price:
                    pnl_pts = -sl_mult * a
                    hit = "SL"
                elif (i - entry_bar) >= horizon:
                    pnl_pts = close[i] - entry_price
                    hit = "TIME"
                else:
                    continue
            else:  # SHORT
                pt_price = entry_price - pt_mult * a
                sl_price = entry_price + sl_mult * a
                if low[i] <= pt_price:
                    pnl_pts = pt_mult * a
                    hit = "PT"
                elif high[i] >= sl_price:
                    pnl_pts = -sl_mult * a
                    hit = "SL"
                elif (i - entry_bar) >= horizon:
                    pnl_pts = -(close[i] - entry_price)
                    hit = "TIME"
                else:
                    continue

            pnl_dollars = pnl_pts * MNQ_PNL_PER_POINT * n_contracts
            daily_pnl[date_str] += pnl_pts
            trades.append({
                "entry_time": idx[entry_bar],
                "exit_time": idx[i],
                "direction": "LONG" if entry_dir == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": close[i] if hit == "TIME" else (pt_price if hit == "PT" else sl_price),
                "pnl_pts": round(pnl_pts, 2),
                "pnl_dollars": round(pnl_dollars, 2),
                "exit_reason": hit,
                "atr_at_entry": round(a, 2),
            })
            in_trade = False
            cooldown = CFG["cooldown_bars"]
            continue

        if cooldown > 0:
            cooldown -= 1
            continue

        # Risk checks
        if daily_pnl[date_str] <= -max_dd:
            continue
        if daily_trades[date_str] >= max_td:
            continue
        if np.isnan(atr[i]) or atr[i] < min_atr:
            continue

        # Signal
        go_long = p_long[i] >= long_th
        go_short = p_short[i] >= short_th

        if go_long and (not go_short or p_long[i] >= p_short[i]):
            in_trade = True
            entry_dir = 1
            entry_price = close[i]
            entry_atr = atr[i]
            entry_bar = i
            daily_trades[date_str] += 1
        elif go_short:
            in_trade = True
            entry_dir = -1
            entry_price = close[i]
            entry_atr = atr[i]
            entry_bar = i
            daily_trades[date_str] += 1

    return pd.DataFrame(trades)


# ── Reporting ─────────────────────────────────────────────────────────────────

def report(trades_df: pd.DataFrame, label: str = "OOS"):
    if trades_df.empty:
        print(f"[{label}] No trades.")
        return

    wins = trades_df[trades_df["pnl_dollars"] > 0]
    losses = trades_df[trades_df["pnl_dollars"] <= 0]
    total_pnl = trades_df["pnl_dollars"].sum()
    n = len(trades_df)
    wr = len(wins) / n
    avg_win = wins["pnl_dollars"].mean() if len(wins) else 0
    avg_loss = losses["pnl_dollars"].mean() if len(losses) else 0

    # Weekly PnL
    trades_df["week"] = pd.to_datetime(trades_df["entry_time"]).dt.to_period("W")
    weekly = trades_df.groupby("week")["pnl_dollars"].sum()

    # Drawdown (cumulative PnL)
    cum = trades_df["pnl_dollars"].cumsum()
    roll_max = cum.cummax()
    dd = (cum - roll_max)
    max_dd = dd.min()

    # Days
    trades_df["date"] = pd.to_datetime(trades_df["entry_time"]).dt.date
    n_days = trades_df["date"].nunique()
    n_weeks = len(weekly)
    trades_per_day = n / n_days

    print(f"\n{'='*60}")
    print(f"  {label} BACKTEST RESULTS  ({n} trades, {n_days} days, {n_weeks} weeks)")
    print(f"{'='*60}")
    print(f"  Total PnL      : ${total_pnl:>10,.0f}")
    print(f"  Weekly avg PnL : ${weekly.mean():>10,.0f}")
    print(f"  Weekly med PnL : ${weekly.median():>10,.0f}")
    print(f"  Weekly best    : ${weekly.max():>10,.0f}")
    print(f"  Weekly worst   : ${weekly.min():>10,.0f}")
    print(f"  Win rate       : {wr*100:>9.1f}%")
    print(f"  Avg win        : ${avg_win:>10,.0f}")
    print(f"  Avg loss       : ${avg_loss:>10,.0f}")
    print(f"  Payoff ratio   : {abs(avg_win/avg_loss if avg_loss else 0):>10.2f}x")
    print(f"  Trades/day     : {trades_per_day:>10.1f}")
    print(f"  Max drawdown   : ${max_dd:>10,.0f}")

    daily = trades_df.groupby("date")["pnl_dollars"].sum()
    print(f"  Avg daily PnL  : ${daily.mean():>10,.0f}")
    print(f"  Daily Sharpe   : {daily.mean()/daily.std()*np.sqrt(252):>10.2f}  (annualized)")
    print(f"\n  By direction:")
    for side in ["LONG", "SHORT"]:
        sub = trades_df[trades_df["direction"] == side]
        if len(sub):
            sw = sub[sub["pnl_dollars"] > 0]
            print(f"    {side}: {len(sub)} trades, WR={len(sw)/len(sub)*100:.1f}%, PnL=${sub['pnl_dollars'].sum():,.0f}")
    print(f"\n  Exit reasons:")
    for reason, grp in trades_df.groupby("exit_reason"):
        print(f"    {reason}: {len(grp)} trades, PnL=${grp['pnl_dollars'].sum():,.0f}")
    print()

    return {
        "total_pnl": round(total_pnl, 2),
        "weekly_avg": round(weekly.mean(), 2),
        "weekly_median": round(weekly.median(), 2),
        "win_rate": round(wr, 4),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "max_drawdown": round(max_dd, 2),
        "trades_per_day": round(trades_per_day, 2),
        "n_trades": n,
        "n_weeks": n_weeks,
        "sharpe_daily": round(daily.mean() / daily.std() * np.sqrt(252), 3) if daily.std() > 0 else 0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    bars_eth = load_bars_5min()
    oos_1min = load_oos_1min_rth()
    ofi = load_ofi()

    # Build features on training data (ETH, Aug-Nov 2025)
    print("\nBuilding features on training + OOS data...")
    all_bars = pd.concat([bars_eth, oos_1min]).sort_index()
    all_bars = all_bars[~all_bars.index.duplicated(keep="last")]
    all_feat = build_features(all_bars, ofi)

    # Split by date
    train_end = pd.Timestamp("2025-11-30", tz="UTC")
    val_end   = pd.Timestamp("2025-12-31", tz="UTC")
    oos_start = pd.Timestamp("2026-01-02", tz="UTC")

    train_df = all_feat[all_feat.index <= train_end]
    val_df   = all_feat[(all_feat.index > train_end) & (all_feat.index <= val_end)]
    oos_df   = all_feat[all_feat.index >= oos_start]

    print(f"\n  Train: {len(train_df)} bars ({train_df.index[0].date()} to {train_df.index[-1].date()})")
    print(f"  Val  : {len(val_df)} bars")
    print(f"  OOS  : {len(oos_df)} bars ({oos_df.index[0].date()} to {oos_df.index[-1].date()})")

    # Label bars for training only (avoid leakage into OOS)
    print("\nLabeling training data...")
    train_labeled = label_bars(train_df)

    # Filter to RTH for labeling quality (approximate: UTC 14-21 = ET 9-16)
    rth_mask = (train_labeled.index.hour >= 14) & (train_labeled.index.hour < 21)
    train_rth = train_labeled[rth_mask]
    print(f"  RTH training samples: {len(train_rth)}")

    # Train models
    print("\nTraining LONG model...")
    long_model = train_model(train_rth, "long")
    print("\nTraining SHORT model...")
    short_model = train_model(train_rth, "short")

    # Build OFI-enriched OOS features
    print("\nBuilding OOS features (with OFI)...")
    oos_feat = build_features(oos_df[["open", "high", "low", "close", "volume"]], ofi)
    oos_labeled = label_bars(oos_feat)  # for reference metrics only

    # Validate on dec 2025
    print("\nValidation (Dec 2025)...")
    val_feat = build_features(val_df[["open", "high", "low", "close", "volume"]], ofi)
    val_labeled = label_bars(val_feat)
    val_rth = val_labeled[(val_labeled.index.hour >= 14) & (val_labeled.index.hour < 21)]
    val_trades = backtest(val_rth, long_model, short_model)
    val_metrics = report(val_trades, "VALIDATION Dec 2025")

    # OOS backtest
    print("\nOOS Backtest (Jan-Mar 2026)...")
    oos_rth = oos_labeled[(oos_labeled.index.hour >= 14) & (oos_labeled.index.hour < 21)]
    oos_trades = backtest(oos_rth, long_model, short_model)
    oos_metrics = report(oos_trades, "OOS Jan-Mar 2026")

    # Feature importance
    import lightgbm as lgb
    print("Top LONG features:")
    imp = pd.Series(
        long_model.feature_importances_,
        index=FEATURE_COLS,
    ).sort_values(ascending=False)
    print(imp.head(10).to_string())

    # Save results
    import json
    results = {
        "config": CFG,
        "validation": val_metrics,
        "oos": oos_metrics,
        "feature_importance_long": imp.to_dict(),
    }
    out_path = RESULTS_DIR / "ml_scalper_v1_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if not oos_trades.empty:
        oos_trades.to_parquet(RESULTS_DIR / "ml_scalper_v1_oos_trades.parquet")
        print(f"Trades saved to {RESULTS_DIR / 'ml_scalper_v1_oos_trades.parquet'}")


if __name__ == "__main__":
    main()
