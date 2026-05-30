"""
ML Scalper v3b — meta-labeling on EMA trend filter.

Approach:
  1. Primary signal: ema9 > ema21 → candidate LONG bar
  2. Triple barrier on those bars only (PT=1.5x, SL=1.0x, horizon=12)
  3. Meta-model: LightGBM trained to predict PT hit | primary fired
  4. Trade when primary fires AND meta p > threshold (~0.50 since base ~45%)

This focuses the model on a pre-filtered universe where trend already agrees,
which should boost WR from ~46% toward 55-60%.

Also: expand train data by including val period (Feb 1-9) since it precedes
the OOS gap (6-week gap, Mar 24 is OOS start).
"""

import base64, gzip, json, pickle, subprocess, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).parents[2]
DATA = BASE / "data" / "processed"
RESULTS = BASE / "ml_intraday_v3" / "results"
RESULTS.mkdir(exist_ok=True)

SSH_HOST = "jg@100.81.204.115"

CFG = dict(
    # Labels (symmetric makes base rate ~45-50%, easier classification)
    pt_atr=1.5, sl_atr=1.0, horizon_bars=12, atr_period=10,
    # Model
    n_estimators=500, learning_rate=0.05, num_leaves=63,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.3, reg_lambda=1.0,
    # Signal — threshold near base rate (meta-label base ~45%)
    long_threshold=0.50, short_threshold=0.99,  # SHORT disabled
    min_atr_pts=5.0,
    # Trade management
    n_contracts=15,
    pt_mult=1.5, sl_mult=1.0,
    max_trades_per_day=6,
    max_daily_loss_pts=133,
    cooldown_bars=3,
)

FEATURE_COLS = [
    # Microstructure (core alpha)
    "ofi_imb", "lg_ofi_imb", "ofi_accel",
    "lg_sm_diverge", "kyles_lambda", "roll_spread",
    "large_frac", "trade_rate_z", "avg_size_z", "max_run_z",
    "ofi_early_n", "ofi_late_n",
    # Technical (context)
    "ret_1", "ret_3", "ret_6", "ret_12",
    "rsi_5", "rsi_14",
    "ema9_ratio", "ema21_ratio", "ema9_21_cross",
    "norm_range", "norm_body",
    "vol_z", "range_pos",
    "vwap_dev",
    # Time
    "hour_sin", "hour_cos", "dow",
    "is_open_30", "is_close_30",
    # Rolling microstructure
    "kyles_lambda_3",
    "lg_ofi_imb_3",
    "trade_rate_z_3",
    # EMA trend (primary signal context)
    "ema_trend",  # 1 if ema9>ema21 else 0
]


def compute_atr(df, period=10):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()


def rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, min_periods=p).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["close"]
    atr = compute_atr(out)
    out["atr"] = atr

    tv = out["total_vol"].replace(0, np.nan)
    out["ofi_early_n"] = out["ofi_early"] / tv.fillna(1)
    out["ofi_late_n"]  = out["ofi_late"]  / tv.fillna(1)

    tr_ma = out["trade_rate"].rolling(20).mean()
    tr_sd = out["trade_rate"].rolling(20).std().replace(0, np.nan)
    out["trade_rate_z"] = (out["trade_rate"] - tr_ma) / tr_sd

    sz_ma = out["avg_size"].rolling(20).mean()
    sz_sd = out["avg_size"].rolling(20).std().replace(0, np.nan)
    out["avg_size_z"] = (out["avg_size"] - sz_ma) / sz_sd

    run_ma = out["max_run"].rolling(20).mean()
    run_sd = out["max_run"].rolling(20).std().replace(0, np.nan)
    out["max_run_z"] = (out["max_run"] - run_ma) / run_sd

    out["kyles_lambda"] = out["kyles_lambda"].clip(0, 2.0)
    out["kyles_lambda_3"]  = out["kyles_lambda"].rolling(3).mean()
    out["lg_ofi_imb_3"]    = out["lg_ofi_imb"].rolling(3).mean()
    out["trade_rate_z_3"]  = out["trade_rate_z"].rolling(3).mean()

    for n in [1, 3, 6, 12]:
        out[f"ret_{n}"] = np.log(c / c.shift(n))

    out["rsi_5"]  = rsi(c, 5)
    out["rsi_14"] = rsi(c, 14)

    ema9  = c.ewm(span=9,  min_periods=9).mean()
    ema21 = c.ewm(span=21, min_periods=21).mean()
    out["ema9_ratio"]    = (c / ema9  - 1) * 100
    out["ema21_ratio"]   = (c / ema21 - 1) * 100
    out["ema9_21_cross"] = (ema9 / ema21 - 1) * 100
    out["ema_trend"]     = (ema9 > ema21).astype(float)  # primary signal

    out["norm_range"] = (out["high"] - out["low"]) / atr
    out["norm_body"]  = (c - out["open"]) / atr.replace(0, np.nan)

    vol_ma = tv.rolling(20).mean()
    vol_sd = tv.rolling(20).std().replace(0, np.nan)
    out["vol_z"] = (tv - vol_ma) / vol_sd

    high20 = out["high"].rolling(20).max()
    low20  = out["low"].rolling(20).min()
    out["range_pos"] = (c - low20) / (high20 - low20).replace(0, np.nan)

    out["vwap_dev"] = (c - out["vwap"]) / atr.replace(0, np.nan)

    h_et = (out.index.hour - 5) % 24
    out["hour_sin"]    = np.sin(2*np.pi*h_et/24)
    out["hour_cos"]    = np.cos(2*np.pi*h_et/24)
    out["dow"]         = out.index.dayofweek.astype(float)
    mins = (h_et - 9)*60 + out.index.minute - 30
    out["is_open_30"]  = (mins <= 30).astype(float)
    out["is_close_30"] = (mins >= 360).astype(float)

    return out


def label_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Label only bars where ema_trend=1 (EMA9>EMA21, primary LONG signal fires)."""
    c   = df["close"].values
    hi  = df["high"].values
    lo  = df["low"].values
    atr = df["atr"].values
    ema_trend = df["ema_trend"].values
    n   = len(df)
    hor = CFG["horizon_bars"]
    ptm = CFG["pt_atr"]
    slm = CFG["sl_atr"]
    ll  = np.full(n, -1, dtype=np.int8)  # -1 = not a primary signal bar

    for i in range(n - hor):
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue
        if ema_trend[i] != 1:
            continue  # meta-labeling: only label when primary fires
        e = c[i]
        l_pt = e + ptm * a
        l_sl = e - slm * a
        ll[i] = 0
        for j in range(i+1, i+hor+1):
            if hi[j] >= l_pt:
                ll[i] = 1
                break
            elif lo[j] <= l_sl:
                break

    df = df.copy()
    df["long_label"] = ll
    return df


REMOTE_SCRIPT = r'''
import sys, gzip, pickle, json, warnings
import numpy as np, pandas as pd
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
    if split < 20:
        raise ValueError(f"Not enough samples: {split}")
    print(f"  [{direction}] n={len(X)} pos={y.mean():.3f}", file=sys.stderr, flush=True)
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
    print(f"  [{direction}] AUC={auc:.4f} iters={model.best_iteration_}", file=sys.stderr, flush=True)
    return model, auc

print("Training LONG (meta-label)...", file=sys.stderr, flush=True)
long_m, long_auc = train_model(train, "long")

def backtest(df, long_m, label=""):
    # Only trade bars where ema_trend=1 (primary signal)
    valid = (df["ema_trend"] == 1) & df[feat_cols].notna().all(axis=1)
    df = df[valid].copy()
    if df.empty: return pd.DataFrame()
    X = df[feat_cols].astype(np.float32)
    df["p_long"] = long_m.predict_proba(X)[:, 1]

    for pct in [50,75,90,95,99]:
        print(f"  [{label}] p_long p{pct}={np.percentile(df['p_long'],pct):.4f}  n_candidates={len(df)}", file=sys.stderr, flush=True)

    trades = []; dp = {}; dtc = {}
    in_trade = False; cooldown = 0
    ep = ea = 0.0; ed = eb = 0
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
        dp.setdefault(ds,0.0); dtc.setdefault(ds,0)
        if in_trade:
            ptp=ep+ptm*ea; slp=ep-slm*ea
            if h[i]>=ptp:   pts,why=ptm*ea,"PT"
            elif l[i]<=slp: pts,why=-slm*ea,"SL"
            elif (i-eb)>=hor: pts,why=c[i]-ep,"TIME"
            else: continue
            dp[ds]+=pts
            trades.append({"entry_time":str(idx[eb]),"exit_time":str(idx[i]),
                           "direction":"LONG",
                           "pnl_pts":round(pts,2),"pnl_dollars":round(pts*scale,2),
                           "exit_reason":why,"atr":round(ea,2)})
            in_trade=False; cooldown=cd; continue
        if cooldown>0: cooldown-=1; continue
        if dp[ds]<=-mdd: continue
        if dtc[ds]>=mtd: continue
        if np.isnan(av[i]) or av[i]<mina: continue
        if pl[i]>=lth:
            in_trade=True; ed=1; ep=c[i]; ea=av[i]; eb=i; dtc[ds]+=1

    return pd.DataFrame(trades)

def summarize(tdf, label):
    if tdf is None or tdf.empty: return {"label":label,"n_trades":0}
    wins=tdf[tdf["pnl_dollars"]>0]; losses=tdf[tdf["pnl_dollars"]<=0]
    tdf["week"]=pd.to_datetime(tdf["entry_time"]).dt.to_period("W").astype(str)
    weekly=tdf.groupby("week")["pnl_dollars"].sum()
    cum=tdf["pnl_dollars"].cumsum(); max_dd=(cum-cum.cummax()).min()
    tdf["date"]=pd.to_datetime(tdf["entry_time"]).dt.strftime("%Y-%m-%d")
    daily=tdf.groupby("date")["pnl_dollars"].sum()
    n=len(tdf); nd=tdf["date"].nunique()
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

print("Val backtest...", file=sys.stderr, flush=True)
vt = backtest(val, long_m, "val")
print("OOS backtest...", file=sys.stderr, flush=True)
ot = backtest(oos, long_m, "oos")

fi = dict(zip(feat_cols, long_m.feature_importances_.tolist()))
print(json.dumps({
    "val":  summarize(vt, "Val Feb 1-9 2026"),
    "oos":  summarize(ot, "OOS Mar-May 2026"),
    "fi":   dict(sorted(fi.items(), key=lambda x:-x[1])[:15]),
    "auc":  {"long": long_auc},
}))
'''


def main():
    print("=== ML Scalper v3b: Meta-labeling on EMA Trend ===", flush=True)

    print("\n[1] Loading microstructure features...", flush=True)
    micro = pd.read_parquet(DATA / "mnq_microstructure_5min.parquet")
    micro.index = pd.to_datetime(micro.index, utc=True)
    print(f"  {len(micro)} bars, {micro.index[0].date()} to {micro.index[-1].date()}", flush=True)

    print("[2] Building features...", flush=True)
    feat = build_features(micro)

    # Expanded train: Dec 2025 + Jan 2026 + Feb 1-9 2026 (all pre-OOS data)
    v_end  = pd.Timestamp("2026-02-09 23:59", tz="UTC")
    o_start= pd.Timestamp("2026-03-24", tz="UTC")

    train = feat[feat.index <= v_end]
    oos   = feat[feat.index >= o_start]
    # Use last 20% of train as pseudo-val for early stopping reporting
    split_ts = train.index[int(len(train)*0.8)]
    val = feat[(feat.index >= split_ts) & (feat.index <= v_end)]

    print(f"  Train (expanded): {len(train)} bars ({train.index[0].date()} - {train.index[-1].date()})", flush=True)
    print(f"  Val (last 20%): {len(val)} bars", flush=True)
    print(f"  OOS: {len(oos)} bars ({oos.index[0].date()} - {oos.index[-1].date()})", flush=True)

    print("[3] Meta-labeling (only EMA9>EMA21 bars)...", flush=True)
    train = label_bars(train)
    val   = label_bars(val)
    oos   = label_bars(oos)
    for name, df in [("train", train), ("val", val), ("oos", oos)]:
        ll = df["long_label"]
        valid = ll[ll >= 0]
        pct = len(valid)/len(df)*100
        print(f"  {name}: {len(valid)} meta-labeled bars ({pct:.1f}% of all), pos_rate={valid.mean():.3f}", flush=True)

    print("\n[4] Serializing payload...", flush=True)
    payload = {"train": train, "val": val, "oos": oos, "cfg": CFG, "feat_cols": FEATURE_COLS}
    data_bytes = gzip.compress(pickle.dumps(payload, protocol=4), compresslevel=1)
    print(f"  Payload: {len(data_bytes)/1e6:.1f} MB", flush=True)

    print(f"\n[5] Remote training on {SSH_HOST}...", flush=True)
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

    print(f"\n  Model AUC — LONG (meta): {results['auc']['long']:.4f}")
    for sk, label in [("val", "VALIDATION"), ("oos", "OOS")]:
        m = results[sk]
        print(f"\n{'='*60}")
        print(f"  {label}: {m['label']}  ({m.get('n_trades',0)} trades, {m.get('n_days',0)} days)")
        print(f"{'='*60}")
        if m.get("n_trades", 0) == 0:
            print("  No trades generated."); continue
        print(f"  Total PnL      : ${m['total_pnl']:>10,.0f}")
        print(f"  Weekly avg     : ${m['weekly_avg']:>10,.0f}   ← target $10k")
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

    print(f"\n  Top features (LONG): {list(results['fi'].keys())[:10]}")

    out_path = RESULTS / "ml_scalper_v3b_results.json"
    save = {k: ({kk:vv for kk,vv in v.items() if kk!="trades"} if isinstance(v,dict) else v)
            for k,v in results.items()}
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nResults saved: {out_path}", flush=True)

    if results["oos"].get("trades"):
        pd.DataFrame(results["oos"]["trades"]).to_parquet(RESULTS / "ml_scalper_v3b_oos_trades.parquet")
        print("OOS trades parquet saved.", flush=True)


if __name__ == "__main__":
    main()
