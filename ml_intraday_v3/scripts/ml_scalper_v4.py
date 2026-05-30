"""
ML Scalper v4 — extended feature set + longer training window.

Changes vs v3:
  - 11 new features: ofi_q1-5 (quintile OFI slices), trade_rate_accel,
    vpin_5, vpin_20, ofi_1m_first_n, ofi_1m_last_n, ofi_15m_n
  - Extended training: Dec 2025 – Feb 28, 2026 (was Dec–Jan only)
  - Same OOS window as v3 (Mar 24+) for direct comparison
  - Same model hyperparams (no overfitting to new features yet)

Data:   mnq_microstructure_5min.parquet (rebuilt with new columns)
W1:     Train Dec 2025 – Feb 28 2026  |  Val Mar 1-23  |  OOS Mar 24 – May 12
W2:     Train Dec 2025 – Mar 31 2026  |  Val Apr 1-6   |  OOS Apr 7 – May 12
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
    # Labels — unchanged from v3
    pt_atr=2.0, sl_atr=1.0, horizon_bars=12, atr_period=10,
    # Model — unchanged from v3
    n_estimators=500, learning_rate=0.03, num_leaves=31,
    min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.5, reg_lambda=2.0,
    # v4 threshold: same as v3 (0.248). Model is more selective — fewer but
    # higher-quality trades. Do NOT lower below 0.240 (WR collapses to 35% below that).
    long_threshold=0.248, short_threshold=0.99,
    min_atr_pts=5.0,
    # Trade management — unchanged from v3
    n_contracts=15,
    pt_mult=2.0, sl_mult=1.0,
    max_trades_per_day=6,
    max_daily_loss_pts=133,
    cooldown_bars=3,
)

# v3 base features
_V3_FEATURES = [
    "ofi_imb", "lg_ofi_imb", "ofi_accel",
    "lg_sm_diverge", "kyles_lambda", "roll_spread",
    "large_frac", "trade_rate_z", "avg_size_z", "max_run_z",
    "ofi_early_n", "ofi_late_n",
    "ret_1", "ret_3", "ret_6", "ret_12",
    "rsi_5", "rsi_14",
    "ema9_ratio", "ema21_ratio", "ema9_21_cross",
    "norm_range", "norm_body",
    "vol_z", "range_pos", "vwap_dev",
    "hour_sin", "hour_cos", "dow",
    "is_open_30", "is_close_30",
    "kyles_lambda_3", "lg_ofi_imb_3", "trade_rate_z_3",
]

# 11 new features added in v4
_V4_NEW_FEATURES = [
    # Sub-bar flow evolution (normalized by total_vol)
    "ofi_q1_n", "ofi_q2_n", "ofi_q3_n", "ofi_q4_n", "ofi_q5_n",
    # Trade burst / urgency
    "trade_rate_accel",
    # Informed-flow toxicity proxy
    "vpin_5", "vpin_20",
    # Cross-timeframe OFI
    "ofi_1m_first_n", "ofi_1m_last_n", "ofi_15m_n",
]

FEATURE_COLS = _V3_FEATURES + _V4_NEW_FEATURES


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
    tv  = out["total_vol"].replace(0, np.nan)

    # ---- v3 features (unchanged) ----
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

    out["kyles_lambda"]    = out["kyles_lambda"].clip(0, 2.0)
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

    out["norm_range"] = (out["high"] - out["low"]) / atr
    out["norm_body"]  = (c - out["open"]) / atr.replace(0, np.nan)

    vol_ma = tv.rolling(20).mean()
    vol_sd = tv.rolling(20).std().replace(0, np.nan)
    out["vol_z"] = (tv - vol_ma) / vol_sd

    high20 = out["high"].rolling(20).max()
    low20  = out["low"].rolling(20).min()
    out["range_pos"] = (c - low20) / (high20 - low20).replace(0, np.nan)
    out["vwap_dev"]  = (c - out["vwap"]) / atr.replace(0, np.nan)

    h_et = (out.index.hour - 5) % 24
    out["hour_sin"]    = np.sin(2*np.pi*h_et/24)
    out["hour_cos"]    = np.cos(2*np.pi*h_et/24)
    out["dow"]         = out.index.dayofweek.astype(float)
    mins = (h_et - 9)*60 + out.index.minute - 30
    out["is_open_30"]  = (mins <= 30).astype(float)
    out["is_close_30"] = (mins >= 360).astype(float)

    # ---- v4 new features ----

    # Quintile OFI: normalize each slice by bar total_vol
    for q in range(1, 6):
        raw = out[f"ofi_q{q}"]
        out[f"ofi_q{q}_n"] = raw / tv.fillna(1)

    # Trade arrival acceleration: winsorize heavy tails
    out["trade_rate_accel"] = out["trade_rate_accel"].clip(-5.0, 5.0)

    # VPIN: already bounded [0, 1] — pass through (clip defensively)
    out["vpin_5"]  = out["vpin_5"].clip(0, 1)
    out["vpin_20"] = out["vpin_20"].clip(0, 1)

    # Cross-timeframe OFI: normalize by total_vol; fill NaN with 0 (missing sub-bar)
    out["ofi_1m_first_n"] = (out["ofi_1m_first"].fillna(0) / tv.fillna(1))
    out["ofi_1m_last_n"]  = (out["ofi_1m_last"].fillna(0)  / tv.fillna(1))

    # 15-min lagged OFI: z-score over rolling 20 bars
    ofi15_ma = out["ofi_15m"].rolling(20).mean()
    ofi15_sd = out["ofi_15m"].rolling(20).std().replace(0, np.nan)
    out["ofi_15m_n"] = ((out["ofi_15m"] - ofi15_ma) / ofi15_sd).clip(-4, 4)

    return out


def label_bars(df: pd.DataFrame) -> pd.DataFrame:
    c    = df["close"].values
    hi   = df["high"].values
    lo   = df["low"].values
    atr  = df["atr"].values
    n    = len(df)
    hor  = CFG["horizon_bars"]
    ptm  = CFG["pt_atr"]
    slm  = CFG["sl_atr"]
    ll   = np.full(n, -1, dtype=np.int8)
    sl   = np.full(n, -1, dtype=np.int8)
    for i in range(n - hor):
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue
        e  = c[i]
        l_pt = e + ptm*a; l_sl = e - slm*a
        s_pt = e - ptm*a; s_sl = e + slm*a
        ll[i] = 0; sl[i] = 0
        for j in range(i+1, i+hor+1):
            if hi[j] >= l_pt: ll[i]=1; break
            elif lo[j] <= l_sl: break
        for j in range(i+1, i+hor+1):
            if lo[j] <= s_pt: sl[i]=1; break
            elif hi[j] >= s_sl: break
    df = df.copy()
    df["long_label"]  = ll
    df["short_label"] = sl
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

print("Training LONG...", file=sys.stderr, flush=True)
long_m,  long_auc  = train_model(train, "long")
print("Training SHORT...", file=sys.stderr, flush=True)
short_m, short_auc = train_model(train, "short")

def backtest(df, long_m, short_m, label=""):
    valid = df[feat_cols].notna().all(axis=1)
    df = df[valid].copy()
    if df.empty: return pd.DataFrame()
    X = df[feat_cols].astype(np.float32)
    df["p_long"]  = long_m.predict_proba(X)[:, 1]
    df["p_short"] = short_m.predict_proba(X)[:, 1]
    for pct in [50,75,90,95,99]:
        print(f"  [{label}] p_long p{pct}={np.percentile(df['p_long'],pct):.4f}  p_short p{pct}={np.percentile(df['p_short'],pct):.4f}", file=sys.stderr, flush=True)

    trades = []; dp = {}; dtc = {}
    in_trade = False; cooldown = 0
    ep = ea = 0.0; ed = eb = 0
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    av = df["atr"].values; pl = df["p_long"].values; ps = df["p_short"].values
    idx = df.index
    nc=cfg["n_contracts"]; ptm=cfg["pt_mult"]; slm=cfg["sl_mult"]
    lth=cfg["long_threshold"]; sth=cfg["short_threshold"]
    mdd=cfg["max_daily_loss_pts"]; mtd=cfg["max_trades_per_day"]
    mina=cfg["min_atr_pts"]; hor=cfg["horizon_bars"]; cd=cfg["cooldown_bars"]
    scale=2.0*nc
    for i in range(len(df)):
        ds = str(idx[i].date())
        dp.setdefault(ds,0.0); dtc.setdefault(ds,0)
        if in_trade:
            a=ea
            if ed==1:
                ptp=ep+ptm*a; slp=ep-slm*a
                if h[i]>=ptp:   pts,why=ptm*a,"PT"
                elif l[i]<=slp: pts,why=-slm*a,"SL"
                elif (i-eb)>=hor: pts,why=c[i]-ep,"TIME"
                else: continue
            else:
                ptp=ep-ptm*a; slp=ep+slm*a
                if l[i]<=ptp:   pts,why=ptm*a,"PT"
                elif h[i]>=slp: pts,why=-slm*a,"SL"
                elif (i-eb)>=hor: pts,why=-(c[i]-ep),"TIME"
                else: continue
            dp[ds]+=pts
            trades.append({"entry_time":str(idx[eb]),"exit_time":str(idx[i]),
                           "direction":"LONG" if ed==1 else "SHORT",
                           "pnl_pts":round(pts,2),"pnl_dollars":round(pts*scale,2),
                           "exit_reason":why,"atr":round(a,2)})
            in_trade=False; cooldown=cd; continue
        if cooldown>0: cooldown-=1; continue
        if dp[ds]<=-mdd: continue
        if dtc[ds]>=mtd: continue
        if np.isnan(av[i]) or av[i]<mina: continue
        h_et = (idx[i].hour - 5) % 24
        if h_et < 11: continue
        if h_et == 13: continue
        if idx[i].weekday() == 3: continue
        gl=pl[i]>=lth; gs=ps[i]>=sth
        if gl and (not gs or pl[i]>=ps[i]):
            in_trade=True;ed=1;ep=c[i];ea=av[i];eb=i;dtc[ds]+=1
        elif gs:
            in_trade=True;ed=-1;ep=c[i];ea=av[i];eb=i;dtc[ds]+=1

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
        "by_direction":{s:{"n":len(sub),"wr":round(len(sub[sub["pnl_dollars"]>0])/len(sub),3),"pnl":round(float(sub["pnl_dollars"].sum()),2)} for s in ["LONG","SHORT"] if len(sub:=tdf[tdf["direction"]==s])>0},
        "by_exit":{r:{"n":len(g),"pnl":round(float(g["pnl_dollars"].sum()),2)} for r,g in tdf.groupby("exit_reason")},
        "trades":tdf.to_dict(orient="records"),
    }

print("Val backtest...", file=sys.stderr, flush=True)
vt = backtest(val,  long_m, short_m, "val")
print("OOS backtest...", file=sys.stderr, flush=True)
ot = backtest(oos,  long_m, short_m, "oos")

import base64
fi = dict(zip(feat_cols, long_m.feature_importances_.tolist()))
model_b64 = base64.b64encode(gzip.compress(pickle.dumps(long_m, protocol=4))).decode()
print(json.dumps({
    "val":  summarize(vt, "Val"),
    "oos":  summarize(ot, "OOS"),
    "fi":   dict(sorted(fi.items(), key=lambda x:-x[1])[:20]),
    "auc":  {"long": long_auc, "short": short_auc},
    "model_b64": model_b64,
}))
'''


def _remote_train(feat, train_end, val_start, val_end, oos_start, window_label):
    t_end   = pd.Timestamp(train_end,  tz="UTC")
    v_start = pd.Timestamp(val_start,  tz="UTC")
    v_end   = pd.Timestamp(val_end,    tz="UTC")
    o_start = pd.Timestamp(oos_start,  tz="UTC")

    train = feat[feat.index <= t_end]
    val   = feat[(feat.index >= v_start) & (feat.index <= v_end)]
    oos   = feat[feat.index >= o_start]

    print(f"\n  [{window_label}] Train: {train.index[0].date()} - {train.index[-1].date()} ({len(train)} bars)", flush=True)
    print(f"  [{window_label}] Val  : {val.index[0].date()} - {val.index[-1].date()} ({len(val)} bars)", flush=True)
    print(f"  [{window_label}] OOS  : {oos.index[0].date()} - {oos.index[-1].date()} ({len(oos)} bars)", flush=True)

    train = label_bars(train)
    val   = label_bars(val)
    oos   = label_bars(oos)
    for name, df in [("train", train), ("val", val), ("oos", oos)]:
        ll = df["long_label"]
        valid = ll[ll >= 0]
        print(f"  [{window_label}] {name}: {len(valid)} valid, pos_rate={valid.mean():.3f}", flush=True)

    # Check new features are present
    missing = [f for f in FEATURE_COLS if f not in feat.columns]
    if missing:
        print(f"  WARNING: {len(missing)} features missing from parquet: {missing}", flush=True)
        print(f"  Falling back to v3 feature set.", flush=True)
        used_cols = [f for f in FEATURE_COLS if f in feat.columns]
    else:
        used_cols = FEATURE_COLS

    payload    = {"train": train, "val": val, "oos": oos, "cfg": CFG, "feat_cols": used_cols}
    data_bytes = gzip.compress(pickle.dumps(payload, protocol=4), compresslevel=1)
    print(f"  [{window_label}] Payload: {len(data_bytes)/1e6:.1f} MB — sending to remote...", flush=True)

    encoded = base64.b64encode(REMOTE_SCRIPT.encode()).decode()
    proc    = subprocess.Popen(
        ["ssh", SSH_HOST, f"python3 -c \"import base64,sys; exec(base64.b64decode('{encoded}').decode())\""],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    stdout_b, stderr_b = proc.communicate(input=data_bytes)
    for line in stderr_b.decode().splitlines():
        print(f"  [{window_label}/remote] {line}", flush=True)
    if proc.returncode != 0:
        print(f"Remote failed ({window_label}):", stderr_b.decode(), flush=True)
        sys.exit(1)

    return json.loads(stdout_b.decode().strip().splitlines()[-1])


def _print_results(results, window_label):
    print(f"\n  [{window_label}] AUC — LONG: {results['auc']['long']:.4f}, SHORT: {results['auc']['short']:.4f}", flush=True)
    for sk, label in [("val", "VALIDATION"), ("oos", "OOS")]:
        m = results[sk]
        print(f"\n{'='*60}")
        print(f"  [{window_label}] {label}: {m['label']}  ({m.get('n_trades',0)} trades, {m.get('n_days',0)} days)")
        print(f"{'='*60}")
        if m.get("n_trades", 0) == 0:
            print("  No trades generated."); continue
        print(f"  Total PnL      : ${m['total_pnl']:>10,.0f}")
        print(f"  Weekly avg     : ${m['weekly_avg']:>10,.0f}   ← target $10k")
        print(f"  Weekly median  : ${m['weekly_median']:>10,.0f}")
        print(f"  Win rate       : {m['win_rate']*100:>9.1f}%")
        print(f"  Avg win/loss   : ${m['avg_win']:,.0f} / ${m['avg_loss']:,.0f}")
        print(f"  Max drawdown   : ${m['max_drawdown']:>10,.0f}")
        print(f"  Trades/day     : {m['trades_per_day']:>10.1f}")
        print(f"  Sharpe (daily) : {m['sharpe_daily']:>10.2f}")
        for s, d in m.get("by_direction", {}).items():
            print(f"    {s}: n={d['n']}, WR={d['wr']*100:.1f}%, PnL=${d['pnl']:,.0f}")
        for r, d in m.get("by_exit", {}).items():
            print(f"    Exit {r}: n={d['n']}, PnL=${d['pnl']:,.0f}")
    print(f"\n  [{window_label}] Top features: {list(results['fi'].keys())[:10]}", flush=True)
    v4_new = [k for k in results['fi'].keys() if k in set(_V4_NEW_FEATURES)]
    if v4_new:
        print(f"  [{window_label}] New v4 features in top-20: {v4_new}", flush=True)


def main():
    print("=== ML Scalper v4: Extended Features + Longer Training ===", flush=True)
    print(f"Features: {len(FEATURE_COLS)} ({len(_V3_FEATURES)} v3 + {len(_V4_NEW_FEATURES)} new)", flush=True)

    print("\n[1] Loading microstructure features...", flush=True)
    micro = pd.read_parquet(DATA / "mnq_microstructure_5min.parquet")
    micro.index = pd.to_datetime(micro.index, utc=True)
    print(f"  {len(micro)} bars, {micro.index[0].date()} to {micro.index[-1].date()}", flush=True)

    # Check if new columns are present (need rebuilt parquet)
    new_cols_present = all(c in micro.columns for c in ["ofi_q1", "trade_rate_accel", "vpin_5", "ofi_1m_first"])
    if not new_cols_present:
        missing = [c for c in ["ofi_q1","trade_rate_accel","vpin_5","ofi_1m_first"] if c not in micro.columns]
        print(f"  WARNING: New raw columns missing: {missing}", flush=True)
        print(f"  Run build_microstructure_features.py first to regenerate the parquet.", flush=True)
        print(f"  Continuing with available features only.", flush=True)
    else:
        print(f"  New columns detected: ofi_q1-5, trade_rate_accel, vpin_5/20, ofi_1m_first/last/15m", flush=True)

    print(f"[2] Building features ({len(FEATURE_COLS)} cols)...", flush=True)
    feat = build_features(micro)

    # W1: same train/val/OOS split as v3 but with new features — direct comparison
    # (Data gap Feb 10 – Mar 23 means train extends only to Feb 9 in practice)
    print("\n[3a] W1: Dec 2025 – Feb 9 2026 train → Mar 24 – May 12 OOS", flush=True)
    r1 = _remote_train(
        feat,
        train_end="2026-02-09 23:59",
        val_start="2026-02-01", val_end="2026-02-09 23:59",
        oos_start="2026-03-24",
        window_label="W1",
    )

    # W2: train includes first week of OOS data as extra context, OOS = Apr–May
    print("\n[3b] W2: Dec 2025 – Mar 31 2026 train → Apr 7 – May 12 OOS", flush=True)
    r2 = _remote_train(
        feat,
        train_end="2026-03-31 23:59",
        val_start="2026-03-24", val_end="2026-04-06 23:59",
        oos_start="2026-04-07",
        window_label="W2",
    )

    MODELS = BASE / "ml_intraday_v3" / "models"
    MODELS.mkdir(exist_ok=True)
    for tag, r in [("w1", r1), ("w2", r2)]:
        if "model_b64" in r:
            model_bytes = gzip.decompress(base64.b64decode(r["model_b64"]))
            path = MODELS / f"ml_scalper_v4_long_{tag}.pkl"
            with open(path, "wb") as f:
                f.write(model_bytes)
            print(f"Model saved ({tag}): {path} ({len(model_bytes)/1e3:.0f} KB)", flush=True)

    if "model_b64" in r1:
        model_bytes = gzip.decompress(base64.b64decode(r1["model_b64"]))
        with open(MODELS / "ml_scalper_v4_long.pkl", "wb") as f:
            f.write(model_bytes)
        print(f"Primary model (W1) → ml_scalper_v4_long.pkl", flush=True)

    _print_results(r1, "W1")
    _print_results(r2, "W2")

    out = {
        "version": "v4",
        "features": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "new_features": _V4_NEW_FEATURES,
        "W1": r1, "W2": r2,
        "config": CFG,
    }
    out_path = RESULTS / "ml_scalper_v4_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
