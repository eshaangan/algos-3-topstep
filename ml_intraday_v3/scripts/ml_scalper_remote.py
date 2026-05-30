"""
ML Scalper v1 — remote execution driver.

Local:  loads data, builds features, serializes to gzip-pickle, pipes to remote via SSH.
Remote: reads from stdin, trains LightGBM, runs OOS backtest, prints JSON to stdout.
Local:  captures stdout and prints results.

Usage:
    python ml_scalper_remote.py
"""

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

CFG = dict(
    pt_atr=2.0,
    sl_atr=1.0,
    horizon_bars=12,
    atr_period=14,
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    long_threshold=0.54,
    short_threshold=0.54,
    min_atr_filter=3.0,
    n_contracts=10,
    pt_mult=2.0,
    sl_mult=1.0,
    max_trades_per_day=8,
    max_daily_loss_pts=200,
    cooldown_bars=2,
)

MNQ_PNL_PER_POINT = 2.0

FEATURE_COLS = [
    "ret_1", "ret_3", "ret_5", "ret_12", "ret_24",
    "rsi_5", "rsi_14",
    "ema9_ratio", "ema21_ratio", "ema9_21_cross", "ema21_50_cross",
    "norm_range", "norm_body", "upper_wick", "lower_wick",
    "vol_zscore", "range_pos",
    "hour_sin", "hour_cos", "dow", "is_first_hour", "is_last_hour",
    "ofi_imb", "ofi_5", "ofi_norm",
]


# ── Local: Data Loading & Feature Engineering ────────────────────────────────

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


def build_features(bars, ofi):
    df = bars.copy()
    close = df["close"]
    atr = compute_atr(df, CFG["atr_period"])
    df["atr"] = atr

    for n in [1, 3, 5, 12, 24]:
        df[f"ret_{n}"] = np.log(close / close.shift(n))

    df["rsi_5"] = rsi(close, 5)
    df["rsi_14"] = rsi(close, 14)

    ema9 = close.ewm(span=9, min_periods=9).mean()
    ema21 = close.ewm(span=21, min_periods=21).mean()
    ema50 = close.ewm(span=50, min_periods=50).mean()
    df["ema9_ratio"] = (close / ema9 - 1) * 100
    df["ema21_ratio"] = (close / ema21 - 1) * 100
    df["ema9_21_cross"] = (ema9 / ema21 - 1) * 100
    df["ema21_50_cross"] = (ema21 / ema50 - 1) * 100

    df["norm_range"] = (df["high"] - df["low"]) / atr
    df["norm_body"] = (close - df["open"]) / atr.replace(0, np.nan)
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / atr.replace(0, np.nan)
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / atr.replace(0, np.nan)

    vol_ma = df["volume"].rolling(20).mean()
    df["vol_zscore"] = (df["volume"] - vol_ma) / df["volume"].rolling(20).std()

    high20 = df["high"].rolling(20).max()
    low20 = df["low"].rolling(20).min()
    df["range_pos"] = (close - low20) / (high20 - low20).replace(0, np.nan)

    # Time — convert UTC index to approximate ET (UTC-5 standard, UTC-4 DST)
    # rough: hour_et = (utc_hour - 5) % 24 gives standard time
    hour_et = (df.index.hour - 5) % 24
    df["hour_sin"] = np.sin(2 * np.pi * hour_et / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour_et / 24)
    df["dow"] = df.index.dayofweek.astype(float)
    df["is_first_hour"] = ((hour_et >= 9) & (hour_et < 10)).astype(float)
    df["is_last_hour"] = ((hour_et >= 15) & (hour_et < 16)).astype(float)

    if ofi is not None and not ofi.empty:
        df = df.join(ofi[["ofi_imb", "ofi", "total_vol"]], how="left")
        df["ofi_imb"] = df["ofi_imb"].fillna(0)
        df["ofi_5"] = df["ofi"].rolling(5).sum().fillna(0)
        tvol5 = df["total_vol"].rolling(5).sum().fillna(0)
        df["ofi_norm"] = np.where(tvol5 > 0, df["ofi_5"] / tvol5, 0.0)
        df.drop(columns=["ofi", "total_vol"], inplace=True, errors="ignore")
    else:
        df["ofi_imb"] = 0.0
        df["ofi_5"] = 0.0
        df["ofi_norm"] = 0.0

    return df


def label_bars(df):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr"].values
    n = len(df)
    horizon = CFG["horizon_bars"]
    pt_mult = CFG["pt_atr"]
    sl_mult = CFG["sl_atr"]

    long_labels = np.zeros(n, dtype=np.int8)
    short_labels = np.zeros(n, dtype=np.int8)

    for i in range(n - horizon):
        a = atr[i]
        if np.isnan(a) or a <= 0:
            long_labels[i] = -1
            short_labels[i] = -1
            continue
        entry = close[i]
        long_pt = entry + pt_mult * a
        long_sl = entry - sl_mult * a
        short_pt = entry - pt_mult * a
        short_sl = entry + sl_mult * a

        for j in range(i + 1, i + horizon + 1):
            if high[j] >= long_pt:
                long_labels[i] = 1
                break
            elif low[j] <= long_sl:
                break

        for j in range(i + 1, i + horizon + 1):
            if low[j] <= short_pt:
                short_labels[i] = 1
                break
            elif high[j] >= short_sl:
                break

    df = df.copy()
    df["long_label"] = long_labels
    df["short_label"] = short_labels
    return df


def load_all():
    print("Loading bars...", flush=True)
    eth = pd.read_hdf(str(DATA / "mnq_aug2025_apr2026_1min_eth.h5"), key="bars_5min_eth")
    eth.index = pd.to_datetime(eth.index, utc=True)

    rth1m = pd.read_hdf(str(DATA / "mnq_2026ytd_1min.h5"), key="bars_1min")
    rth1m.index = pd.to_datetime(rth1m.index, utc=True)
    rth5m = rth1m.resample("5min").agg({"open": "first", "high": "max", "low": "min",
                                         "close": "last", "volume": "sum"}).dropna()

    print("Loading OFI...", flush=True)
    ofi_parts = []
    for f in sorted(OFI_DIR.glob("*.parquet")):
        tmp = pd.read_parquet(f)
        tmp.index = pd.to_datetime(tmp.index, utc=True)
        ofi_parts.append(tmp)
    ofi_raw = pd.concat(ofi_parts).sort_index()
    ofi_raw = ofi_raw[~ofi_raw.index.duplicated(keep="last")]
    ofi5 = ofi_raw.resample("5min").agg({"buy_vol": "sum", "sell_vol": "sum",
                                          "ofi": "sum", "total_vol": "sum", "n_trades": "sum"})
    ofi5["ofi_imb"] = ofi5["ofi"] / ofi5["total_vol"].replace(0, np.nan)

    # Merge and deduplicate bars
    all_bars = pd.concat([eth, rth5m]).sort_index()
    all_bars = all_bars[~all_bars.index.duplicated(keep="last")]

    print("Building features...", flush=True)
    feat = build_features(all_bars, ofi5)

    # Splits — train on Aug-Nov 2025, val Dec 2025, OOS Jan-Apr 2026
    train_end = pd.Timestamp("2025-11-30", tz="UTC")
    val_end   = pd.Timestamp("2025-12-31", tz="UTC")
    oos_start = pd.Timestamp("2026-01-02", tz="UTC")

    def rth_mask(df):
        # UTC 14:30-21:00 = ET 9:30-16:00 (standard) / 13:30-20:00 (DST)
        return (df.index.hour >= 14) & (df.index.hour < 21)

    train_df = feat[feat.index <= train_end]
    val_df   = feat[(feat.index > train_end) & (feat.index <= val_end)]
    oos_df   = feat[feat.index >= oos_start]

    train_rth = train_df[rth_mask(train_df)]
    val_rth   = val_df[rth_mask(val_df)]
    oos_rth   = oos_df[rth_mask(oos_df)]

    print(f"  Train RTH: {len(train_rth)} bars", flush=True)
    print(f"  Val   RTH: {len(val_rth)} bars", flush=True)
    print(f"  OOS   RTH: {len(oos_rth)} bars ({oos_rth.index[0].date()} - {oos_rth.index[-1].date()})", flush=True)

    print("Labeling...", flush=True)
    train_labeled = label_bars(train_rth)
    val_labeled   = label_bars(val_rth)
    oos_labeled   = label_bars(oos_rth)

    return train_labeled, val_labeled, oos_labeled


# ── Remote script (runs on SSH host, reads pickled DataFrames from stdin) ────

REMOTE_SCRIPT = r'''
import sys, gzip, pickle, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# Read pickled payload from stdin
payload = pickle.loads(gzip.decompress(sys.stdin.buffer.read()))
train = payload["train"]
val   = payload["val"]
oos   = payload["oos"]
cfg   = payload["cfg"]
feat_cols = payload["feat_cols"]

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

def train_model(df, direction):
    label_col = f"{direction}_label"
    mask = (df[label_col] >= 0) & df[feat_cols].notna().all(axis=1)
    X = df.loc[mask, feat_cols].astype(np.float32)
    y = df.loc[mask, label_col].astype(np.int32)
    split = int(len(X) * 0.8)
    model = lgb.LGBMClassifier(
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        num_leaves=cfg["num_leaves"],
        min_child_samples=cfg["min_child_samples"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        is_unbalance=True,
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
    print(f"  [{direction}] n={len(X)}, pos_rate={y.mean():.3f}, val_AUC={auc:.4f}, iters={model.best_iteration_}", file=sys.stderr, flush=True)
    return model

print("Training LONG...", file=sys.stderr, flush=True)
long_m = train_model(train, "long")
print("Training SHORT...", file=sys.stderr, flush=True)
short_m = train_model(train, "short")

def backtest(df, long_model, short_model, label=""):
    valid = df[feat_cols].notna().all(axis=1)
    df = df[valid].copy()
    X = df[feat_cols].astype(np.float32)
    df["p_long"] = long_model.predict_proba(X)[:, 1]
    df["p_short"] = short_model.predict_proba(X)[:, 1]

    trades = []
    daily_pnl = {}
    daily_tc = {}
    in_trade = False
    cooldown = 0
    entry_price = entry_atr = 0.0
    entry_dir = entry_bar = 0

    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    atr_v = df["atr"].values
    p_long  = df["p_long"].values
    p_short = df["p_short"].values
    idx = df.index

    nc   = cfg["n_contracts"]
    ptm  = cfg["pt_mult"]
    slm  = cfg["sl_mult"]
    lth  = cfg["long_threshold"]
    sth  = cfg["short_threshold"]
    mdd  = cfg["max_daily_loss_pts"]
    mtd  = cfg["max_trades_per_day"]
    mina = cfg["min_atr_filter"]
    hor  = cfg["horizon_bars"]
    cd   = cfg["cooldown_bars"]
    pnl_scale = 2.0 * nc  # $ per MNQ point

    for i in range(len(df)):
        ds = str(idx[i].date())
        daily_pnl.setdefault(ds, 0.0)
        daily_tc.setdefault(ds, 0)

        if in_trade:
            a = entry_atr
            if entry_dir == 1:
                pt_p = entry_price + ptm * a
                sl_p = entry_price - slm * a
                if high[i] >= pt_p: pnl_pts, hit = ptm * a, "PT"
                elif low[i] <= sl_p: pnl_pts, hit = -slm * a, "SL"
                elif (i - entry_bar) >= hor: pnl_pts, hit = close[i] - entry_price, "TIME"
                else: continue
            else:
                pt_p = entry_price - ptm * a
                sl_p = entry_price + slm * a
                if low[i] <= pt_p: pnl_pts, hit = ptm * a, "PT"
                elif high[i] >= sl_p: pnl_pts, hit = -slm * a, "SL"
                elif (i - entry_bar) >= hor: pnl_pts, hit = -(close[i] - entry_price), "TIME"
                else: continue

            pnl_d = pnl_pts * pnl_scale
            daily_pnl[ds] += pnl_pts
            trades.append({
                "entry_time": str(idx[entry_bar]),
                "exit_time": str(idx[i]),
                "direction": "LONG" if entry_dir == 1 else "SHORT",
                "pnl_pts": round(pnl_pts, 2),
                "pnl_dollars": round(pnl_d, 2),
                "exit_reason": hit,
                "atr": round(a, 2),
            })
            in_trade = False
            cooldown = cd
            continue

        if cooldown > 0: cooldown -= 1; continue
        if daily_pnl[ds] <= -mdd: continue
        if daily_tc[ds] >= mtd: continue
        if np.isnan(atr_v[i]) or atr_v[i] < mina: continue

        go_long  = p_long[i]  >= lth
        go_short = p_short[i] >= sth

        if go_long and (not go_short or p_long[i] >= p_short[i]):
            in_trade = True; entry_dir = 1
            entry_price = close[i]; entry_atr = atr_v[i]; entry_bar = i
            daily_tc[ds] += 1
        elif go_short:
            in_trade = True; entry_dir = -1
            entry_price = close[i]; entry_atr = atr_v[i]; entry_bar = i
            daily_tc[ds] += 1

    return pd.DataFrame(trades)

def summarize(trades_df, label):
    if trades_df.empty:
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
    wr = len(wins) / n
    return {
        "label": label,
        "n_trades": n,
        "n_days": trades_df["date"].nunique(),
        "trades_per_day": round(n / trades_df["date"].nunique(), 2),
        "total_pnl": round(trades_df["pnl_dollars"].sum(), 2),
        "weekly_avg": round(weekly.mean(), 2),
        "weekly_median": round(float(weekly.median()), 2),
        "weekly_best": round(weekly.max(), 2),
        "weekly_worst": round(weekly.min(), 2),
        "win_rate": round(wr, 4),
        "avg_win": round(wins["pnl_dollars"].mean(), 2) if len(wins) else 0,
        "avg_loss": round(losses["pnl_dollars"].mean(), 2) if len(losses) else 0,
        "max_drawdown": round(max_dd, 2),
        "sharpe_daily": round(daily.mean() / daily.std() * (252**0.5), 3) if daily.std() > 0 else 0,
        "by_direction": {
            side: {
                "n": len(sub),
                "wr": round(len(sub[sub["pnl_dollars"]>0])/len(sub), 3),
                "pnl": round(sub["pnl_dollars"].sum(), 2),
            }
            for side in ["LONG","SHORT"]
            if len(sub := trades_df[trades_df["direction"]==side]) > 0
        },
        "by_exit": {
            r: {"n": len(g), "pnl": round(g["pnl_dollars"].sum(), 2)}
            for r, g in trades_df.groupby("exit_reason")
        },
        "trades": trades_df.to_dict(orient="records"),
    }

print("Val backtest...", file=sys.stderr, flush=True)
val_trades = backtest(val, long_m, short_m, "val")
print("OOS backtest...", file=sys.stderr, flush=True)
oos_trades = backtest(oos, long_m, short_m, "oos")

# Feature importance
fi = dict(zip(feat_cols, long_m.feature_importances_.tolist()))

results = {
    "val":  summarize(val_trades,  "Dec 2025 Validation"),
    "oos":  summarize(oos_trades,  "Jan-Apr 2026 OOS"),
    "feature_importance_long": dict(sorted(fi.items(), key=lambda x: -x[1])[:15]),
    "cfg": cfg,
}
print(json.dumps(results))
'''


def main():
    train_df, val_df, oos_df = load_all()

    print("Serializing payload...", flush=True)
    payload = {
        "train": train_df,
        "val": val_df,
        "oos": oos_df,
        "cfg": CFG,
        "feat_cols": FEATURE_COLS,
    }
    data_bytes = gzip.compress(pickle.dumps(payload, protocol=4), compresslevel=1)
    print(f"  Payload size: {len(data_bytes)/1e6:.1f} MB", flush=True)

    print(f"Launching remote computation on {SSH_HOST}...", flush=True)
    import base64
    encoded = base64.b64encode(REMOTE_SCRIPT.encode()).decode()
    remote_cmd = f"python3 -c \"import base64,sys; exec(base64.b64decode('{encoded}').decode())\""
    proc = subprocess.Popen(
        ["ssh", SSH_HOST, remote_cmd],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = proc.communicate(input=data_bytes)

    # Remote stderr = progress logs
    if stderr_bytes:
        for line in stderr_bytes.decode().splitlines():
            print(f"  [remote] {line}", flush=True)

    if proc.returncode != 0:
        print(f"Remote failed (exit {proc.returncode}):", flush=True)
        print(stderr_bytes.decode(), flush=True)
        sys.exit(1)

    # Parse JSON from stdout
    results = json.loads(stdout_bytes.decode().strip().splitlines()[-1])

    # Pretty print
    for split_key in ["val", "oos"]:
        m = results[split_key]
        label = m["label"]
        print(f"\n{'='*60}")
        print(f"  {label}  ({m['n_trades']} trades, {m.get('n_days',0)} days)")
        print(f"{'='*60}")
        print(f"  Total PnL      : ${m.get('total_pnl',0):>10,.0f}")
        print(f"  Weekly avg     : ${m.get('weekly_avg',0):>10,.0f}")
        print(f"  Weekly median  : ${m.get('weekly_median',0):>10,.0f}")
        print(f"  Weekly best    : ${m.get('weekly_best',0):>10,.0f}")
        print(f"  Weekly worst   : ${m.get('weekly_worst',0):>10,.0f}")
        print(f"  Win rate       : {m.get('win_rate',0)*100:>9.1f}%")
        print(f"  Avg win/loss   : ${m.get('avg_win',0):,.0f} / ${m.get('avg_loss',0):,.0f}")
        print(f"  Max drawdown   : ${m.get('max_drawdown',0):>10,.0f}")
        print(f"  Trades/day     : {m.get('trades_per_day',0):>10.1f}")
        print(f"  Sharpe (daily) : {m.get('sharpe_daily',0):>10.2f}")
        if "by_direction" in m:
            for side, d in m["by_direction"].items():
                print(f"    {side}: n={d['n']}, WR={d['wr']*100:.1f}%, PnL=${d['pnl']:,.0f}")
        if "by_exit" in m:
            for reason, d in m["by_exit"].items():
                print(f"    Exit {reason}: n={d['n']}, PnL=${d['pnl']:,.0f}")

    print(f"\n  Top LONG features: {list(results['feature_importance_long'].keys())[:8]}")

    # Save
    out = RESULTS_DIR / "ml_scalper_v1_results.json"
    with open(out, "w") as f:
        # Don't save full trades list to keep file small
        save = {k: v for k, v in results.items() if k != "trades"}
        for split_key in ["val", "oos"]:
            if split_key in save and "trades" in save[split_key]:
                del save[split_key]["trades"]
        json.dump(save, f, indent=2)
    print(f"\nResults saved to {out}", flush=True)

    # Save OOS trades parquet locally
    if results["oos"].get("trades"):
        oos_trades_df = pd.DataFrame(results["oos"]["trades"])
        oos_trades_df.to_parquet(RESULTS_DIR / "ml_scalper_v1_oos_trades.parquet")
        print(f"OOS trades saved ({len(oos_trades_df)} rows)", flush=True)


if __name__ == "__main__":
    main()
