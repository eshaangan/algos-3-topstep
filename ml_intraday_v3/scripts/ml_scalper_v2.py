"""
ML Scalper v2 — 1-min bars, OFI-enriched, remote training via SSH.

Training data: Dec 2025 1-min OHLCV (built from trade ticks) + Jan-Feb 2026 1-min RTH
OOS test    : Mar-Apr 2026 1-min RTH (from mnq_2026ytd_1min.h5)
OFI         : available from Dec 2025, fully used in training

Key improvements over v1:
  - 1-min resolution (more samples, microstructure signal)
  - OFI features present during training (not all-zero)
  - Lower confidence threshold (0.52)
  - More training data: Dec 2025 + Jan-Feb 2026 (~30k RTH bars)
  - Tighter PT/SL: 1.5x/1.0x ATR (easier to win, more trades)
"""

import base64
import gzip
import io
import json
import pickle
import subprocess
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

SSH_HOST = "jg@100.81.204.115"
PRICE_SCALE = 1e9

CFG = dict(
    # Labels: simple 30-min forward return direction (pos_rate ≈ 50%)
    fwd_bars=30,         # 30-min forward return
    atr_period=14,
    # Model: logistic regression + LGBM ensemble
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=15,       # shallow trees — less overfit
    min_child_samples=200,  # very conservative
    subsample=0.7,
    colsample_bytree=0.6,
    reg_alpha=1.0,       # L1 regularization
    reg_lambda=5.0,      # L2 regularization
    # Signal
    long_threshold=0.52,
    short_threshold=0.52,
    min_atr_pts=2.0,
    # Trade management
    n_contracts=10,
    pt_atr=2.5,
    sl_atr=1.5,
    horizon_bars=30,
    max_trades_per_day=6,
    max_daily_loss_pts=200,  # 200 pts × $2 × 10c = $4,000
    cooldown_bars=15,
)

MNQ_PNL_PER_POINT = 2.0

FEATURE_COLS = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    "rsi_5", "rsi_14",
    "ema9_ratio", "ema21_ratio", "ema9_21_cross",
    "norm_range", "norm_body", "upper_wick", "lower_wick",
    "vol_zscore",
    "range_pos_20",
    "vwap_dev",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "dow", "is_open_30", "is_close_30",
    "ofi_imb", "ofi_3", "ofi_5", "ofi_norm",
    "ofi_accel",
]


# ── Data Building ─────────────────────────────────────────────────────────────

def build_dec2025_1min_rth() -> pd.DataFrame:
    """Build Dec 2025 1-min OHLCV from trade CSV (RTH only)."""
    path = DATA / "mnq_trades_dec2025.csv.gz"
    print("  Building Dec 2025 1-min bars from ticks...", flush=True)

    chunks = []
    for chunk in pd.read_csv(
        path,
        usecols=["ts_recv", "price", "size"],
        dtype={"price": "float64", "size": "float64"},
        compression=None,   # file is plain CSV despite .gz extension
        chunksize=2_000_000,
    ):
        chunk = chunk.dropna()
        chunk["ts"] = pd.to_datetime(chunk["ts_recv"], unit="ns", utc=True).dt.floor("1min")
        chunk["price_f"] = chunk["price"] / PRICE_SCALE
        # RTH: 14:30-20:59 UTC (9:30-15:59 ET standard time)
        h = chunk["ts"].dt.hour
        m = chunk["ts"].dt.minute
        in_rth = (h > 14) | ((h == 14) & (m >= 30))
        in_rth &= (h < 21)
        chunk = chunk[in_rth]
        if chunk.empty:
            continue
        agg = chunk.groupby("ts").agg(
            open=("price_f", "first"),
            high=("price_f", "max"),
            low=("price_f", "min"),
            close=("price_f", "last"),
            volume=("size", "sum"),
        )
        chunks.append(agg)

    df = pd.concat(chunks).groupby(level=0).agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    })
    print(f"  Dec 2025 1-min RTH: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}", flush=True)
    return df


def load_2026_1min_rth() -> pd.DataFrame:
    """Load Jan-Mar 2026 1-min RTH bars."""
    df = pd.read_hdf(str(DATA / "mnq_2026ytd_1min.h5"), key="bars_1min")
    df.index = pd.to_datetime(df.index, utc=True)
    print(f"  2026 1-min RTH: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}", flush=True)
    return df


def load_ofi_1min() -> pd.DataFrame:
    """Load and concatenate all 1-min OFI parquets."""
    parts = []
    for f in sorted(OFI_DIR.glob("*.parquet")):
        tmp = pd.read_parquet(f)
        tmp.index = pd.to_datetime(tmp.index, utc=True)
        parts.append(tmp)
    ofi = pd.concat(parts).sort_index()
    ofi = ofi[~ofi.index.duplicated(keep="last")]
    ofi["ofi_imb"] = ofi["ofi"] / ofi["total_vol"].replace(0, np.nan).fillna(0)
    print(f"  OFI 1-min: {len(ofi)} bars, {ofi.index[0].date()} to {ofi.index[-1].date()}", flush=True)
    return ofi


# ── Feature Engineering ───────────────────────────────────────────────────────

def compute_atr(df, period=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features(bars: pd.DataFrame, ofi: pd.DataFrame) -> pd.DataFrame:
    df = bars[["open", "high", "low", "close", "volume"]].copy()
    close = df["close"]
    atr = compute_atr(df, CFG["atr_period"])
    df["atr"] = atr

    for n in [1, 3, 5, 10, 20]:
        df[f"ret_{n}"] = np.log(close / close.shift(n))

    df["rsi_5"]  = rsi(close, 5)
    df["rsi_14"] = rsi(close, 14)

    ema9  = close.ewm(span=9,  min_periods=9).mean()
    ema21 = close.ewm(span=21, min_periods=21).mean()
    df["ema9_ratio"]    = (close / ema9  - 1) * 100
    df["ema21_ratio"]   = (close / ema21 - 1) * 100
    df["ema9_21_cross"] = (ema9  / ema21 - 1) * 100

    df["norm_range"]  = (df["high"] - df["low"]) / atr
    df["norm_body"]   = (close - df["open"]) / atr.replace(0, np.nan)
    hi_oc = df[["open", "close"]].max(axis=1)
    lo_oc = df[["open", "close"]].min(axis=1)
    df["upper_wick"]  = (df["high"] - hi_oc) / atr.replace(0, np.nan)
    df["lower_wick"]  = (lo_oc - df["low"]) / atr.replace(0, np.nan)

    vol_ma = df["volume"].rolling(20).mean()
    df["vol_zscore"] = (df["volume"] - vol_ma) / df["volume"].rolling(20).std().replace(0, np.nan)

    high20 = df["high"].rolling(20).max()
    low20  = df["low"].rolling(20).min()
    df["range_pos_20"] = (close - low20) / (high20 - low20).replace(0, np.nan)

    # Intraday VWAP deviation (reset each day)
    df["_dv"] = close * df["volume"]
    df["_day"] = df.index.date
    vwap = df.groupby("_day")[["_dv", "volume"]].cumsum()
    df["vwap"] = vwap["_dv"] / vwap["volume"].replace(0, np.nan)
    df["vwap_dev"] = (close - df["vwap"]) / atr.replace(0, np.nan)
    df.drop(columns=["_dv", "_day", "vwap"], inplace=True)

    # Time features
    hour_et = (df.index.hour - 5) % 24
    minute  = df.index.minute
    mins_from_open = (hour_et - 9) * 60 + minute - 30
    df["hour_sin"]   = np.sin(2 * np.pi * hour_et / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * hour_et / 24)
    df["minute_sin"] = np.sin(2 * np.pi * minute / 60)
    df["minute_cos"] = np.cos(2 * np.pi * minute / 60)
    df["dow"]        = df.index.dayofweek.astype(float)
    df["is_open_30"] = (mins_from_open <= 30).astype(float)
    df["is_close_30"] = (mins_from_open >= 360).astype(float)

    # OFI features (aligned by timestamp)
    df = df.join(ofi[["ofi_imb", "ofi", "total_vol"]], how="left")
    df["ofi_imb"]  = df["ofi_imb"].fillna(0)
    df["ofi_3"]    = df["ofi"].rolling(3).sum().fillna(0)
    df["ofi_5"]    = df["ofi"].rolling(5).sum().fillna(0)
    tvol5 = df["total_vol"].rolling(5).sum().fillna(0)
    df["ofi_norm"] = np.where(tvol5 > 0, df["ofi_5"] / tvol5, 0.0)
    df["ofi_accel"] = df["ofi_imb"].diff().fillna(0)
    df.drop(columns=["ofi", "total_vol"], inplace=True, errors="ignore")

    return df


# ── Triple-Barrier Labeling ───────────────────────────────────────────────────

def label_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple directional labels:
      long_label  = 1 if close[i + fwd_bars] > close[i]  (buy signal)
      short_label = 1 if close[i + fwd_bars] < close[i]  (sell signal)
    Note: long_label + short_label = 1 always (they're inverses),
    but we train two separate models so the threshold can be tuned independently.
    """
    fwd = CFG["fwd_bars"]
    close = df["close"].values
    n = len(df)
    long_labels  = np.full(n, -1, dtype=np.int8)
    short_labels = np.full(n, -1, dtype=np.int8)
    for i in range(n - fwd):
        if np.isnan(close[i]) or np.isnan(close[i + fwd]):
            continue
        up = int(close[i + fwd] > close[i])
        long_labels[i]  = up
        short_labels[i] = 1 - up
    df = df.copy()
    df["long_label"]  = long_labels
    df["short_label"] = short_labels
    return df


# ── Remote script ─────────────────────────────────────────────────────────────

REMOTE_SCRIPT = r'''
import sys, gzip, pickle, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

payload   = pickle.loads(gzip.decompress(sys.stdin.buffer.read()))
train     = payload["train"]
val       = payload["val"]
oos       = payload["oos"]
cfg       = payload["cfg"]
feat_cols = payload["feat_cols"]

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

def train_model(df, direction):
    lc = f"{direction}_label"
    mask = (df[lc] >= 0) & df[feat_cols].notna().all(axis=1)
    X = df.loc[mask, feat_cols].astype(np.float32)
    y = df.loc[mask, lc].astype(np.int32)
    split = int(len(X) * 0.8)
    print(f"  [{direction}] n={len(X)}, pos={y.mean():.3f}, split={split}", file=sys.stderr, flush=True)
    if split < 10:
        raise ValueError(f"Too few training samples: {split}")
    model = lgb.LGBMClassifier(
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        num_leaves=cfg["num_leaves"],
        min_child_samples=cfg["min_child_samples"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        reg_alpha=cfg.get("reg_alpha", 0.0),
        reg_lambda=cfg.get("reg_lambda", 0.0),
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X.iloc[:split], y.iloc[:split],
        eval_set=[(X.iloc[split:], y.iloc[split:])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    auc = roc_auc_score(y.iloc[split:], model.predict_proba(X.iloc[split:])[:, 1])
    print(f"  [{direction}] AUC={auc:.4f} iters={model.best_iteration_}", file=sys.stderr, flush=True)
    return model

print("Training LONG...", file=sys.stderr, flush=True)
long_m  = train_model(train, "long")
print("Training SHORT...", file=sys.stderr, flush=True)
short_m = train_model(train, "short")

def backtest(df, long_m, short_m):
    valid = df[feat_cols].notna().all(axis=1)
    df = df[valid].copy()
    if df.empty:
        return pd.DataFrame()
    X = df[feat_cols].astype(np.float32)
    df["p_long"]  = long_m.predict_proba(X)[:, 1]
    df["p_short"] = short_m.predict_proba(X)[:, 1]

    trades = []
    daily_pnl = {}; daily_tc = {}
    in_trade = False; cooldown = 0
    entry_price = entry_atr = 0.0
    entry_dir = entry_bar = 0

    c = df["close"].values; h = df["high"].values; l = df["low"].values
    atr_v = df["atr"].values
    pl = df["p_long"].values; ps = df["p_short"].values
    idx = df.index

    nc = cfg["n_contracts"]; ptm = cfg["pt_atr"]; slm = cfg["sl_atr"]
    lth = cfg["long_threshold"]; sth = cfg["short_threshold"]
    mdd = cfg["max_daily_loss_pts"]; mtd = cfg["max_trades_per_day"]
    mina = cfg["min_atr_pts"]; hor = cfg["horizon_bars"]; cd = cfg["cooldown_bars"]
    scale = 2.0 * nc

    for i in range(len(df)):
        ds = str(idx[i].date())
        daily_pnl.setdefault(ds, 0.0); daily_tc.setdefault(ds, 0)

        if in_trade:
            a = entry_atr
            if entry_dir == 1:
                ptp = entry_price + ptm*a; slp = entry_price - slm*a
                if h[i] >= ptp:            pts, why = ptm*a,  "PT"
                elif l[i] <= slp:          pts, why = -slm*a, "SL"
                elif (i - entry_bar) >= hor: pts, why = c[i]-entry_price, "TIME"
                else: continue
            else:
                ptp = entry_price - ptm*a; slp = entry_price + slm*a
                if l[i] <= ptp:            pts, why = ptm*a,  "PT"
                elif h[i] >= slp:          pts, why = -slm*a, "SL"
                elif (i - entry_bar) >= hor: pts, why = -(c[i]-entry_price), "TIME"
                else: continue
            daily_pnl[ds] += pts
            trades.append({"entry_time": str(idx[entry_bar]), "exit_time": str(idx[i]),
                           "direction": "LONG" if entry_dir==1 else "SHORT",
                           "pnl_pts": round(pts,2), "pnl_dollars": round(pts*scale,2),
                           "exit_reason": why, "atr": round(a,2)})
            in_trade = False; cooldown = cd
            continue

        if cooldown > 0: cooldown -= 1; continue
        if daily_pnl[ds] <= -mdd: continue
        if daily_tc[ds] >= mtd: continue
        if np.isnan(atr_v[i]) or atr_v[i] < mina: continue

        go_long  = pl[i] >= lth
        go_short = ps[i] >= sth
        if go_long and (not go_short or pl[i] >= ps[i]):
            in_trade = True; entry_dir = 1; entry_price = c[i]; entry_atr = atr_v[i]; entry_bar = i; daily_tc[ds] += 1
        elif go_short:
            in_trade = True; entry_dir = -1; entry_price = c[i]; entry_atr = atr_v[i]; entry_bar = i; daily_tc[ds] += 1

    return pd.DataFrame(trades)

def summarize(trades_df, label):
    if trades_df is None or trades_df.empty:
        return {"label": label, "n_trades": 0}
    wins = trades_df[trades_df["pnl_dollars"] > 0]
    losses = trades_df[trades_df["pnl_dollars"] <= 0]
    trades_df["week"] = pd.to_datetime(trades_df["entry_time"]).dt.to_period("W")
    weekly = trades_df.groupby("week")["pnl_dollars"].sum()
    cum = trades_df["pnl_dollars"].cumsum()
    max_dd = (cum - cum.cummax()).min()
    trades_df["date"] = pd.to_datetime(trades_df["entry_time"]).dt.date
    daily = trades_df.groupby("date")["pnl_dollars"].sum()
    n = len(trades_df)
    return {
        "label": label, "n_trades": n,
        "n_days": int(trades_df["date"].nunique()),
        "trades_per_day": round(n / max(trades_df["date"].nunique(), 1), 2),
        "total_pnl": round(float(trades_df["pnl_dollars"].sum()), 2),
        "weekly_avg": round(float(weekly.mean()), 2),
        "weekly_median": round(float(weekly.median()), 2),
        "weekly_best": round(float(weekly.max()), 2),
        "weekly_worst": round(float(weekly.min()), 2),
        "win_rate": round(len(wins)/n, 4),
        "avg_win": round(float(wins["pnl_dollars"].mean()), 2) if len(wins) else 0,
        "avg_loss": round(float(losses["pnl_dollars"].mean()), 2) if len(losses) else 0,
        "max_drawdown": round(float(max_dd), 2),
        "sharpe_daily": round(float(daily.mean()/daily.std()*(252**0.5)), 3) if daily.std() > 0 else 0,
        "by_direction": {
            s: {"n": len(sub), "wr": round(len(sub[sub["pnl_dollars"]>0])/len(sub),3), "pnl": round(float(sub["pnl_dollars"].sum()),2)}
            for s in ["LONG","SHORT"]
            if len(sub := trades_df[trades_df["direction"]==s]) > 0
        },
        "by_exit": {r: {"n": len(g), "pnl": round(float(g["pnl_dollars"].sum()),2)} for r,g in trades_df.groupby("exit_reason")},
        "trades": trades_df.to_dict(orient="records"),
    }

print("Val backtest...", file=sys.stderr, flush=True)
val_t = backtest(val, long_m, short_m)
print("OOS backtest...", file=sys.stderr, flush=True)
oos_t = backtest(oos, long_m, short_m)

fi = dict(zip(feat_cols, long_m.feature_importances_.tolist()))
results = {
    "val": summarize(val_t,  "Val (Feb 2026)"),
    "oos": summarize(oos_t,  "OOS (Mar-Apr 2026)"),
    "feature_importance_long": dict(sorted(fi.items(), key=lambda x: -x[1])[:15]),
    "cfg": cfg,
}
print(json.dumps(results))
'''


def main():
    print("=== ML Scalper v2: 1-min bars, OFI-enriched ===", flush=True)

    # Build training bars
    print("\n[1] Building bars...", flush=True)
    dec_bars  = build_dec2025_1min_rth()
    bars_2026 = load_2026_1min_rth()
    ofi       = load_ofi_1min()

    # Combine all bars
    all_bars = pd.concat([dec_bars, bars_2026]).sort_index()
    all_bars = all_bars[~all_bars.index.duplicated(keep="last")]
    print(f"  Combined: {len(all_bars)} 1-min bars ({all_bars.index[0].date()} to {all_bars.index[-1].date()})", flush=True)

    # Features
    print("\n[2] Building features...", flush=True)
    feat = build_features(all_bars, ofi)

    # Splits
    # Train: Dec 2025 + Jan 2026
    # Val  : Feb 2026
    # OOS  : Mar 2026 + (any Apr if present)
    train_end = pd.Timestamp("2026-01-31 23:59", tz="UTC")
    val_end   = pd.Timestamp("2026-02-28 23:59", tz="UTC")
    oos_start = pd.Timestamp("2026-03-01", tz="UTC")

    train_df = feat[feat.index <= train_end]
    val_df   = feat[(feat.index > train_end) & (feat.index <= val_end)]
    oos_df   = feat[feat.index >= oos_start]

    print(f"  Train: {len(train_df)} bars ({train_df.index[0].date()} to {train_df.index[-1].date()})", flush=True)
    print(f"  Val  : {len(val_df)} bars", flush=True)
    print(f"  OOS  : {len(oos_df)} bars ({oos_df.index[0].date()} to {oos_df.index[-1].date()})", flush=True)

    # Label
    print("\n[3] Labeling...", flush=True)
    train_labeled = label_bars(train_df)
    val_labeled   = label_bars(val_df)
    oos_labeled   = label_bars(oos_df)

    for split_name, df in [("train", train_labeled), ("val", val_labeled), ("oos", oos_labeled)]:
        ll = df["long_label"]
        print(f"  {split_name}: {(ll>=0).sum()} valid, pos_rate={ll[ll>=0].mean():.3f}", flush=True)

    # Serialize
    print("\n[4] Serializing payload...", flush=True)
    payload = {
        "train": train_labeled,
        "val":   val_labeled,
        "oos":   oos_labeled,
        "cfg":   CFG,
        "feat_cols": FEATURE_COLS,
    }
    data_bytes = gzip.compress(pickle.dumps(payload, protocol=4), compresslevel=1)
    print(f"  Size: {len(data_bytes)/1e6:.1f} MB", flush=True)

    # Run remotely
    print(f"\n[5] Remote training on {SSH_HOST}...", flush=True)
    encoded = base64.b64encode(REMOTE_SCRIPT.encode()).decode()
    remote_cmd = f"python3 -c \"import base64,sys; exec(base64.b64decode('{encoded}').decode())\""
    proc = subprocess.Popen(
        ["ssh", SSH_HOST, remote_cmd],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = proc.communicate(input=data_bytes)

    for line in stderr_bytes.decode().splitlines():
        print(f"  [remote] {line}", flush=True)

    if proc.returncode != 0:
        print(f"Remote failed (exit {proc.returncode})", flush=True)
        sys.exit(1)

    # Parse results
    raw_lines = stdout_bytes.decode().strip().splitlines()
    results = json.loads(raw_lines[-1])

    # Print results
    for split_key, label in [("val", "VALIDATION Feb 2026"), ("oos", "OOS Mar-Apr 2026")]:
        m = results[split_key]
        print(f"\n{'='*60}")
        print(f"  {label}  ({m.get('n_trades',0)} trades, {m.get('n_days',0)} days)")
        print(f"{'='*60}")
        if m.get("n_trades", 0) == 0:
            print("  No trades generated.")
            continue
        print(f"  Total PnL      : ${m['total_pnl']:>10,.0f}")
        print(f"  Weekly avg     : ${m['weekly_avg']:>10,.0f}")
        print(f"  Weekly median  : ${m['weekly_median']:>10,.0f}")
        print(f"  Weekly best    : ${m['weekly_best']:>10,.0f}")
        print(f"  Weekly worst   : ${m['weekly_worst']:>10,.0f}")
        print(f"  Win rate       : {m['win_rate']*100:>9.1f}%")
        print(f"  Avg win / loss : ${m['avg_win']:,.0f} / ${m['avg_loss']:,.0f}")
        print(f"  Max drawdown   : ${m['max_drawdown']:>10,.0f}")
        print(f"  Trades/day     : {m['trades_per_day']:>10.1f}")
        print(f"  Sharpe (daily) : {m['sharpe_daily']:>10.2f}")
        for side, d in m.get("by_direction", {}).items():
            print(f"    {side}: n={d['n']}, WR={d['wr']*100:.1f}%, PnL=${d['pnl']:,.0f}")
        for reason, d in m.get("by_exit", {}).items():
            print(f"    Exit {reason}: n={d['n']}, PnL=${d['pnl']:,.0f}")

    top_fi = list(results["feature_importance_long"].keys())[:8]
    print(f"\n  Top LONG features: {top_fi}")

    # Save
    out_path = RESULTS_DIR / "ml_scalper_v2_results.json"
    save = {k: {kk: vv for kk, vv in v.items() if kk != "trades"} if isinstance(v, dict) else v
            for k, v in results.items()}
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nResults saved: {out_path}", flush=True)

    if results["oos"].get("trades"):
        oos_df_out = pd.DataFrame(results["oos"]["trades"])
        oos_df_out.to_parquet(RESULTS_DIR / "ml_scalper_v2_oos_trades.parquet")
        print(f"OOS trades saved: {len(oos_df_out)} rows", flush=True)


if __name__ == "__main__":
    main()
