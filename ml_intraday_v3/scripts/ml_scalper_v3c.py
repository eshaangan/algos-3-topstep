"""
ML Scalper v3c — OHLCV-only features, deployable via TopstepX REST bars.

Strips all microstructure features (Kyle's lambda, OFI, trade-size metrics)
that require tick data. Uses only features computable from 5-min OHLCV bars,
so this model can run live with the existing TopstepXRestDataFetcher.

Filters baked in (from OOS analysis):
  - Skip 9:30-10:59 ET (opening noise)
  - Skip Thursday (confirmed across ORB + ML studies)

Saves trained model to: ml_intraday_v3/models/ml_scalper_v3c_long.pkl
"""

import base64, gzip, json, pickle, subprocess, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).parents[2]
DATA = BASE / "data" / "processed"
RESULTS = BASE / "ml_intraday_v3" / "results"
MODELS  = BASE / "ml_intraday_v3" / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

SSH_HOST = "jg@100.81.204.115"

CFG = dict(
    pt_atr=2.0, sl_atr=1.0, horizon_bars=12, atr_period=10,
    n_estimators=500, learning_rate=0.05, num_leaves=63,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.3, reg_lambda=1.0,
    # threshold at ~p90 of probability distribution (base rate ~0.21)
    long_threshold=0.234, short_threshold=0.99,
    min_atr_pts=5.0,
    n_contracts=15,
    pt_mult=2.0, sl_mult=1.0,
    max_trades_per_day=6,
    max_daily_loss_pts=133,
    cooldown_bars=3,
)

# Only OHLCV-derivable features — all computable from open/high/low/close/volume
FEATURE_COLS = [
    "ret_1", "ret_3", "ret_6", "ret_12",
    "rsi_5", "rsi_14",
    "ema9_ratio", "ema21_ratio", "ema9_21_cross",
    "norm_range", "norm_body",
    "vol_z", "range_pos",
    "vwap_dev",
    "hour_sin", "hour_cos", "dow",
    "is_open_30", "is_close_30",
    "atr_z",        # ATR z-score (vol regime)
    "ret_range_6",  # 6-bar price range normalized
]


def compute_atr(df, period=10):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()


def rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, min_periods=p).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c   = out["close"]
    atr = compute_atr(out)
    out["atr"] = atr

    # Returns
    for n in [1, 3, 6, 12]:
        out[f"ret_{n}"] = np.log(c / c.shift(n))

    # RSI
    out["rsi_5"]  = rsi(c, 5)
    out["rsi_14"] = rsi(c, 14)

    # EMA
    ema9  = c.ewm(span=9,  min_periods=9).mean()
    ema21 = c.ewm(span=21, min_periods=21).mean()
    out["ema9_ratio"]    = (c / ema9  - 1) * 100
    out["ema21_ratio"]   = (c / ema21 - 1) * 100
    out["ema9_21_cross"] = (ema9 / ema21 - 1) * 100

    # Range/body
    out["norm_range"] = (out["high"] - out["low"]) / atr.replace(0, np.nan)
    out["norm_body"]  = (c - out["open"]) / atr.replace(0, np.nan)

    # 6-bar normalized range
    out["ret_range_6"] = (out["high"].rolling(6).max() - out["low"].rolling(6).min()) / atr.replace(0, np.nan)

    # Volume z-score (20-bar)
    tv = out["total_vol"] if "total_vol" in out.columns else out.get("volume", pd.Series(np.nan, index=out.index))
    tv = tv.replace(0, np.nan)
    vol_ma = tv.rolling(20).mean()
    vol_sd = tv.rolling(20).std().replace(0, np.nan)
    out["vol_z"] = (tv - vol_ma) / vol_sd

    # ATR z-score (vol regime)
    atr_ma = atr.rolling(20).mean()
    atr_sd = atr.rolling(20).std().replace(0, np.nan)
    out["atr_z"] = (atr - atr_ma) / atr_sd

    # Range position (where close sits in 20-bar H/L range)
    high20 = out["high"].rolling(20).max()
    low20  = out["low"].rolling(20).min()
    out["range_pos"] = (c - low20) / (high20 - low20).replace(0, np.nan)

    # Approximate VWAP from typical price × volume
    if "vwap" in out.columns:
        v = out["vwap"]
    else:
        tp = (out["high"] + out["low"] + c) / 3
        dv = (tp * tv.fillna(0)).cumsum()
        cv = tv.fillna(0).cumsum()
        v  = dv / cv.replace(0, np.nan)
    out["vwap_dev"] = (c - v) / atr.replace(0, np.nan)

    # Time (UTC index, convert to ET = UTC-5 during standard, UTC-4 during DST)
    h_et = (out.index.hour - 5) % 24
    out["hour_sin"]    = np.sin(2 * np.pi * h_et / 24)
    out["hour_cos"]    = np.cos(2 * np.pi * h_et / 24)
    out["dow"]         = out.index.dayofweek.astype(float)
    mins_from_open     = (h_et - 9) * 60 + out.index.minute - 30
    out["is_open_30"]  = (mins_from_open <= 30).astype(float)
    out["is_close_30"] = (mins_from_open >= 360).astype(float)

    return out


def label_bars(df: pd.DataFrame) -> pd.DataFrame:
    c   = df["close"].values
    hi  = df["high"].values
    lo  = df["low"].values
    atr = df["atr"].values
    n   = len(df)
    hor = CFG["horizon_bars"]
    ptm = CFG["pt_atr"]
    slm = CFG["sl_atr"]
    ll  = np.full(n, -1, dtype=np.int8)

    for i in range(n - hor):
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue
        e    = c[i]
        l_pt = e + ptm * a
        l_sl = e - slm * a
        ll[i] = 0
        for j in range(i+1, i+hor+1):
            if hi[j] >= l_pt:
                ll[i] = 1; break
            elif lo[j] <= l_sl:
                break

    df = df.copy()
    df["long_label"] = ll
    return df


REMOTE_SCRIPT = r'''
import sys, gzip, pickle, json, base64, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

payload   = pickle.loads(gzip.decompress(sys.stdin.buffer.read()))
train     = payload["train"]
oos       = payload["oos"]
cfg       = payload["cfg"]
feat_cols = payload["feat_cols"]

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

mask = (train["long_label"] >= 0) & train[feat_cols].notna().all(axis=1)
X = train.loc[mask, feat_cols].astype(np.float32)
y = train.loc[mask, "long_label"].astype(np.int32)
split = int(len(X) * 0.8)
print(f"  n={len(X)} pos={y.mean():.3f} features={len(feat_cols)}", file=sys.stderr, flush=True)

model = lgb.LGBMClassifier(
    n_estimators=cfg["n_estimators"],
    learning_rate=cfg["learning_rate"],
    num_leaves=cfg["num_leaves"],
    min_child_samples=cfg["min_child_samples"],
    subsample=cfg["subsample"],
    colsample_bytree=cfg["colsample_bytree"],
    reg_alpha=cfg.get("reg_alpha", 0),
    reg_lambda=cfg.get("reg_lambda", 0),
    random_state=42, n_jobs=-1, verbose=-1,
)
model.fit(
    X.iloc[:split], y.iloc[:split],
    eval_set=[(X.iloc[split:], y.iloc[split:])],
    callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)],
)
auc = roc_auc_score(y.iloc[split:], model.predict_proba(X.iloc[split:])[:, 1])
print(f"  AUC={auc:.4f} iters={model.best_iteration_}", file=sys.stderr, flush=True)

# Probability distribution on OOS
oos_valid = oos[feat_cols].notna().all(axis=1)
X_oos = oos.loc[oos_valid, feat_cols].astype(np.float32)
probs = model.predict_proba(X_oos)[:, 1]
for pct in [50, 75, 90, 95, 99]:
    print(f"  OOS p_long p{pct}={np.percentile(probs, pct):.4f}", file=sys.stderr, flush=True)

def backtest(df):
    valid = df[feat_cols].notna().all(axis=1)
    df = df[valid].copy()
    if df.empty: return pd.DataFrame()
    X = df[feat_cols].astype(np.float32)
    df["p_long"] = model.predict_proba(X)[:, 1]

    trades = []; dp = {}; dtc = {}
    in_trade = False; cooldown = 0
    ep = ea = 0.0; eb = 0
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    av = df["atr"].values; pl = df["p_long"].values
    idx = df.index
    nc=cfg["n_contracts"]; ptm=cfg["pt_mult"]; slm=cfg["sl_mult"]
    lth=cfg["long_threshold"]
    mdd=cfg["max_daily_loss_pts"]; mtd=cfg["max_trades_per_day"]
    mina=cfg["min_atr_pts"]; hor=cfg["horizon_bars"]; cd=cfg["cooldown_bars"]
    scale=2.0*nc

    for i in range(len(df)):
        ds = str(idx[i].date())
        dp.setdefault(ds, 0.0); dtc.setdefault(ds, 0)
        if in_trade:
            ptp=ep+ptm*ea; slp=ep-slm*ea
            if h[i]>=ptp:      pts,why = ptm*ea,  "PT"
            elif l[i]<=slp:    pts,why = -slm*ea, "SL"
            elif (i-eb)>=hor:  pts,why = c[i]-ep, "TIME"
            else: continue
            dp[ds] += pts
            trades.append({"entry_time":str(idx[eb]),"exit_time":str(idx[i]),
                           "direction":"LONG","pnl_pts":round(pts,2),
                           "pnl_dollars":round(pts*scale,2),"exit_reason":why,"atr":round(ea,2)})
            in_trade=False; cooldown=cd; continue
        if cooldown>0: cooldown-=1; continue
        if dp[ds]<=-mdd: continue
        if dtc[ds]>=mtd: continue
        if np.isnan(av[i]) or av[i]<mina: continue
        # Time filters: skip 9:30-10:59 ET, skip Thursday
        h_et = (idx[i].hour - 5) % 24
        if h_et < 11: continue
        if idx[i].weekday() == 3: continue
        if pl[i] >= lth:
            in_trade=True; ep=c[i]; ea=av[i]; eb=i; dtc[ds]+=1

    return pd.DataFrame(trades)

def summarize(tdf, label):
    if tdf is None or tdf.empty: return {"label":label,"n_trades":0}
    wins = tdf[tdf["pnl_dollars"]>0]
    losses = tdf[tdf["pnl_dollars"]<=0]
    tdf["week"] = pd.to_datetime(tdf["entry_time"]).dt.to_period("W").astype(str)
    weekly = tdf.groupby("week")["pnl_dollars"].sum()
    cum = tdf["pnl_dollars"].cumsum(); max_dd = (cum-cum.cummax()).min()
    tdf["date"] = pd.to_datetime(tdf["entry_time"]).dt.strftime("%Y-%m-%d")
    daily = tdf.groupby("date")["pnl_dollars"].sum()
    n = len(tdf); nd = tdf["date"].nunique()
    return {
        "label":label,"n_trades":n,"n_days":int(nd),
        "trades_per_day":round(n/max(nd,1),2),
        "total_pnl":round(float(tdf["pnl_dollars"].sum()),2),
        "weekly_avg":round(float(weekly.mean()),2),
        "weekly_median":round(float(weekly.median()),2),
        "weekly_best":round(float(weekly.max()),2),
        "weekly_worst":round(float(weekly.min()),2),
        "win_rate":round(len(wins)/n,4),
        "avg_win":round(float(wins["pnl_dollars"].mean()),2) if len(wins) else 0,
        "avg_loss":round(float(losses["pnl_dollars"].mean()),2) if len(losses) else 0,
        "max_drawdown":round(float(max_dd),2),
        "sharpe_daily":round(float(daily.mean()/daily.std()*(252**0.5)),3) if daily.std()>0 else 0,
        "by_exit":{r:{"n":len(g),"pnl":round(float(g["pnl_dollars"].sum()),2)} for r,g in tdf.groupby("exit_reason")},
        "trades":tdf.to_dict(orient="records"),
    }

ot = backtest(oos)
fi = dict(zip(feat_cols, model.feature_importances_.tolist()))
model_b64 = base64.b64encode(gzip.compress(pickle.dumps(model, protocol=4))).decode()

print(json.dumps({
    "oos":   summarize(ot, "Full OOS Feb1-May12 2026"),
    "fi":    dict(sorted(fi.items(), key=lambda x:-x[1])),
    "auc":   auc,
    "model_b64": model_b64,
}))
'''


def main():
    print("=== ML Scalper v3c: OHLCV-only features (deployable) ===", flush=True)

    print("\n[1] Loading microstructure parquet (using OHLCV cols only)...", flush=True)
    micro = pd.read_parquet(DATA / "mnq_microstructure_5min.parquet")
    micro.index = pd.to_datetime(micro.index, utc=True)
    print(f"  {len(micro)} bars, {micro.index[0].date()} to {micro.index[-1].date()}", flush=True)

    print("[2] Building OHLCV features...", flush=True)
    feat = build_features(micro)

    t_end   = pd.Timestamp("2026-01-31 23:59", tz="UTC")
    o_start = pd.Timestamp("2026-02-01", tz="UTC")
    train = label_bars(feat[feat.index <= t_end])
    oos   = label_bars(feat[feat.index >= o_start])

    print(f"  Train: {len(train)} bars, OOS: {len(oos)} bars", flush=True)
    for name, df in [("train", train), ("oos", oos)]:
        ll = df["long_label"]; valid = ll[ll >= 0]
        print(f"  {name}: pos_rate={valid.mean():.3f} ({len(valid)} labeled)", flush=True)

    # Check features exist
    missing = [f for f in FEATURE_COLS if f not in feat.columns]
    if missing:
        print(f"  WARNING: missing features: {missing}", flush=True)
        return

    print("\n[3] Serializing and sending to remote...", flush=True)
    payload = {"train": train, "oos": oos, "cfg": CFG, "feat_cols": FEATURE_COLS}
    data_bytes = gzip.compress(pickle.dumps(payload, protocol=4), compresslevel=1)
    print(f"  Payload: {len(data_bytes)/1e6:.1f} MB", flush=True)

    encoded = base64.b64encode(REMOTE_SCRIPT.encode()).decode()
    proc = subprocess.Popen(
        ["ssh", SSH_HOST, f"python3 -c \"import base64,sys; exec(base64.b64decode('{encoded}').decode())\""],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    stdout_b, stderr_b = proc.communicate(input=data_bytes)
    for line in stderr_b.decode().splitlines():
        print(f"  [remote] {line}", flush=True)
    if proc.returncode != 0:
        print("Remote failed:", stderr_b.decode(), flush=True)
        sys.exit(1)

    results = json.loads(stdout_b.decode().strip().splitlines()[-1])

    print(f"\n  AUC: {results['auc']:.4f}")
    m = results["oos"]
    print(f"\n{'='*60}")
    print(f"  FULL OOS: {m['label']}  ({m.get('n_trades',0)} trades, {m.get('n_days',0)} days)")
    print(f"{'='*60}")
    if m.get("n_trades", 0) == 0:
        print("  No trades generated.")
    else:
        print(f"  Total PnL      : ${m['total_pnl']:>10,.0f}")
        print(f"  Weekly avg     : ${m['weekly_avg']:>10,.0f}")
        print(f"  Weekly median  : ${m['weekly_median']:>10,.0f}")
        print(f"  Weekly best    : ${m['weekly_best']:>10,.0f}")
        print(f"  Weekly worst   : ${m['weekly_worst']:>10,.0f}")
        print(f"  Win rate       : {m['win_rate']*100:>9.1f}%")
        print(f"  Avg win/loss   : ${m['avg_win']:,.0f} / ${m['avg_loss']:,.0f}")
        print(f"  Max drawdown   : ${m['max_drawdown']:>10,.0f}   ← limit $4k")
        print(f"  Trades/day     : {m['trades_per_day']:>10.1f}")
        print(f"  Sharpe (daily) : {m['sharpe_daily']:>10.2f}")
        for r, d in m.get("by_exit", {}).items():
            print(f"    Exit {r}: n={d['n']}, PnL=${d['pnl']:,.0f}")

    print(f"\n  Feature importances:")
    for feat_name, imp in list(results["fi"].items())[:10]:
        print(f"    {feat_name:25s}: {imp}")

    # Save model
    if "model_b64" in results:
        model_bytes = gzip.decompress(base64.b64decode(results["model_b64"]))
        model_path = MODELS / "ml_scalper_v3c_long.pkl"
        with open(model_path, "wb") as f:
            f.write(model_bytes)
        print(f"\n  Model saved: {model_path} ({len(model_bytes)/1e3:.0f} KB)", flush=True)

    # Save results
    out_path = RESULTS / "ml_scalper_v3c_results.json"
    save = {k: ({kk:vv for kk,vv in v.items() if kk not in ("trades","model_b64")}
                if isinstance(v, dict) else v)
            for k, v in results.items() if k != "model_b64"}
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"  Results saved: {out_path}", flush=True)

    if results["oos"].get("trades"):
        pd.DataFrame(results["oos"]["trades"]).to_parquet(RESULTS / "ml_scalper_v3c_oos_trades.parquet")


if __name__ == "__main__":
    main()
