"""
ML Scalper v5 — class-balanced + probability-calibrated.

Key changes vs v3:
  1. scale_pos_weight=4.5  → fixes class imbalance (18% positive labels)
                            → probability distribution shifts right
                            → threshold fires at ~p75 instead of p95
                            → 2-3× more trades at modest WR cost
  2. Dynamic threshold     → set from val p75 after training (not hardcoded)
  3. Platt scaling         → calibrate probabilities so EV threshold is meaningful
  4. Relaxed regularisation→ reg_alpha=0.3, reg_lambda=1.5 (model was over-regularised)

Requires: 3+ years of tick data (MotiveWave export → build_microstructure_features.py)
Will fall back to 6-month dataset with a warning if full data not yet available.

Usage:
    python ml_intraday_v3/scripts/ml_scalper_v5.py
"""

import base64, gzip, json, pickle, subprocess, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE    = Path(__file__).parents[2]
DATA    = BASE / "data" / "processed"
RESULTS = BASE / "ml_intraday_v3" / "results"
RESULTS.mkdir(exist_ok=True)

SSH_HOST = "jg@100.81.204.115"

CFG = dict(
    # Labels — run barrier_grid_search.py first to confirm best values
    pt_atr=1.5, sl_atr=1.0, horizon_bars=12, atr_period=10,

    # Model — scale_pos_weight is the key change
    n_estimators=600, learning_rate=0.02, num_leaves=31,
    min_child_samples=40, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.3, reg_lambda=1.5,
    scale_pos_weight=4.5,    # 82/18 ≈ 4.5; corrects class imbalance

    # Threshold: set dynamically from val distribution (see THRESHOLD_PERCENTILE)
    # long_threshold is overwritten after training
    long_threshold=None, short_threshold=0.99,
    threshold_pct=75,    # use val p75 as the firing threshold
    min_atr_pts=5.0,

    # Trade management (Lucid 100k)
    n_contracts=6,
    pt_mult=1.5, sl_mult=1.0,
    max_trades_per_day=4,
    max_daily_loss_pts=100,  # 100pts × $2 × 6c = $1,200 internal limit
    cooldown_bars=2,
)

FEATURE_COLS = [
    # Microstructure (core alpha) — same as v3
    "ofi_imb", "lg_ofi_imb", "ofi_accel",
    "lg_sm_diverge", "kyles_lambda", "roll_spread",
    "large_frac", "trade_rate_z", "avg_size_z", "max_run_z",
    "ofi_early_n", "ofi_late_n",
    # Technical
    "ret_1", "ret_3", "ret_6", "ret_12",
    "rsi_5", "rsi_14",
    "ema9_ratio", "ema21_ratio", "ema9_21_cross",
    "norm_range", "norm_body", "vol_z", "range_pos", "vwap_dev",
    # Time
    "hour_sin", "hour_cos", "dow", "is_open_30", "is_close_30",
    # Rolling microstructure
    "kyles_lambda_3", "lg_ofi_imb_3", "trade_rate_z_3",
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
    tv  = out["total_vol"].replace(0, np.nan)

    out["ofi_early_n"] = out["ofi_early"] / tv.fillna(1)
    out["ofi_late_n"]  = out["ofi_late"]  / tv.fillna(1)

    for col, suffix in [("trade_rate","z"), ("avg_size","z"), ("max_run","z")]:
        ma = out[col].rolling(20).mean()
        sd = out[col].rolling(20).std().replace(0, np.nan)
        out[f"{col}_{suffix}"] = (out[col] - ma) / sd

    out["kyles_lambda"]   = out["kyles_lambda"].clip(0, 2.0)
    out["kyles_lambda_3"] = out["kyles_lambda"].rolling(3).mean()
    out["lg_ofi_imb_3"]   = out["lg_ofi_imb"].rolling(3).mean()
    out["trade_rate_z_3"] = out["trade_rate_z"].rolling(3).mean()

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
    out["hour_sin"]   = np.sin(2*np.pi*h_et/24)
    out["hour_cos"]   = np.cos(2*np.pi*h_et/24)
    out["dow"]        = out.index.dayofweek.astype(float)
    mins = (h_et - 9)*60 + out.index.minute - 30
    out["is_open_30"]  = (mins <= 30).astype(float)
    out["is_close_30"] = (mins >= 360).astype(float)
    return out


def label_bars(df: pd.DataFrame) -> pd.DataFrame:
    c   = df["close"].values; hi = df["high"].values; lo = df["low"].values
    atr = df["atr"].values;   n  = len(df)
    hor = CFG["horizon_bars"];ptm = CFG["pt_atr"]; slm = CFG["sl_atr"]
    ll = np.full(n, -1, dtype=np.int8)
    sl = np.full(n, -1, dtype=np.int8)
    for i in range(n - hor):
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue
        e = c[i]
        l_pt = e + ptm*a; l_sl = e - slm*a
        s_pt = e - ptm*a; s_sl = e + slm*a
        ll[i] = sl[i] = 0
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
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

def train_model(df, direction):
    lc = f"{direction}_label"
    mask = (df[lc] >= 0) & df[feat_cols].notna().all(axis=1)
    X = df.loc[mask, feat_cols].astype(np.float32)
    y = df.loc[mask, lc].astype(np.int32)
    pos_rate = float(y.mean())
    split = int(len(X) * 0.8)
    if split < 20:
        raise ValueError(f"Not enough samples: {split}")
    print(f"  [{direction}] n={len(X)} pos_rate={pos_rate:.3f} "
          f"scale_pos_weight={cfg.get('scale_pos_weight', 1.0):.1f}",
          file=sys.stderr, flush=True)

    model = lgb.LGBMClassifier(
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        num_leaves=cfg["num_leaves"],
        min_child_samples=cfg["min_child_samples"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        reg_alpha=cfg.get("reg_alpha", 0),
        reg_lambda=cfg.get("reg_lambda", 0),
        scale_pos_weight=cfg.get("scale_pos_weight", 1.0),
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(
        X.iloc[:split], y.iloc[:split],
        eval_set=[(X.iloc[split:], y.iloc[split:])],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)],
    )
    auc = roc_auc_score(y.iloc[split:], model.predict_proba(X.iloc[split:])[:, 1])
    print(f"  [{direction}] AUC={auc:.4f} iters={model.best_iteration_}",
          file=sys.stderr, flush=True)
    return model, auc

# --- Platt scaling calibration ---
def calibrate(model, X_val, y_val):
    """Fit isotonic calibration on val set."""
    raw_proba = model.predict_proba(X_val)[:, 1]
    # Platt scaling: logistic regression on logit(raw)
    logit = np.log(raw_proba.clip(1e-6, 1-1e-6) / (1 - raw_proba.clip(1e-6, 1-1e-6)))
    lr = LogisticRegression(C=1e4)
    lr.fit(logit.reshape(-1, 1), y_val)
    def calibrated_predict_proba(X):
        raw = model.predict_proba(X)[:, 1]
        logit = np.log(raw.clip(1e-6, 1-1e-6) / (1 - raw.clip(1e-6, 1-1e-6)))
        return lr.predict_proba(logit.reshape(-1, 1))[:, 1]
    return calibrated_predict_proba, lr

print("Training LONG...", file=sys.stderr, flush=True)
long_m, long_auc = train_model(train, "long")

# Calibrate on val
val_mask = (val["long_label"] >= 0) & val[feat_cols].notna().all(axis=1)
X_val = val.loc[val_mask, feat_cols].astype(np.float32)
y_val = val.loc[val_mask, "long_label"].astype(np.int32)
cal_fn, cal_lr = calibrate(long_m, X_val, y_val)

# Compute val probability percentiles (raw and calibrated)
val_all = val[val[feat_cols].notna().all(axis=1)]
X_val_all = val_all[feat_cols].astype(np.float32)
p_raw = long_m.predict_proba(X_val_all)[:, 1]
p_cal = cal_fn(X_val_all)

thresh_pct = cfg.get("threshold_pct", 75)
raw_thresh  = float(np.percentile(p_raw, thresh_pct))
cal_thresh  = float(np.percentile(p_cal, thresh_pct))

print(f"  [threshold] Raw p{thresh_pct}={raw_thresh:.4f}  Calibrated p{thresh_pct}={cal_thresh:.4f}",
      file=sys.stderr, flush=True)
for pct in [50, 75, 90, 95, 99]:
    print(f"  [val dist] raw p{pct}={np.percentile(p_raw, pct):.4f}  "
          f"cal p{pct}={np.percentile(p_cal, pct):.4f}", file=sys.stderr, flush=True)

print("Training SHORT...", file=sys.stderr, flush=True)
short_m, short_auc = train_model(train, "short")

def backtest(df, long_m, short_m, cal_fn, label="", use_calibrated=True):
    valid = df[feat_cols].notna().all(axis=1)
    df = df[valid].copy()
    if df.empty: return pd.DataFrame()
    X = df[feat_cols].astype(np.float32)

    if use_calibrated:
        df["p_long"]  = cal_fn(X)
        lth = cal_thresh
    else:
        df["p_long"]  = long_m.predict_proba(X)[:, 1]
        lth = raw_thresh

    df["p_short"] = short_m.predict_proba(X)[:, 1]
    sth = cfg["short_threshold"]

    for pct in [50, 75, 90, 95, 99]:
        print(f"  [{label}] p_long p{pct}={np.percentile(df['p_long'],pct):.4f}",
              file=sys.stderr, flush=True)

    trades = []; dp = {}; dtc = {}
    in_trade = False; cooldown = 0
    ep = ea = 0.0; ed = eb = 0
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    av = df["atr"].values; pl = df["p_long"].values; ps = df["p_short"].values
    idx = df.index
    nc=cfg["n_contracts"]; ptm=cfg["pt_mult"]; slm=cfg["sl_mult"]
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
        "win_rate":round(len(wins)/n,4),
        "avg_win":round(float(wins["pnl_dollars"].mean()),2) if len(wins) else 0,
        "avg_loss":round(float(losses["pnl_dollars"].mean()),2) if len(losses) else 0,
        "max_drawdown":round(float(max_dd),2),
        "sharpe_daily":round(float(daily.mean()/daily.std()*(252**0.5)),3) if daily.std()>0 else 0,
        "by_direction":{s:{"n":len(sub),"wr":round(len(sub[sub["pnl_dollars"]>0])/len(sub),3),"pnl":round(float(sub["pnl_dollars"].sum()),2)} for s in ["LONG","SHORT"] if len(sub:=tdf[tdf["direction"]==s])>0},
        "by_exit":{r:{"n":len(g),"pnl":round(float(g["pnl_dollars"].sum()),2)} for r,g in tdf.groupby("exit_reason")},
        "trades":tdf.to_dict(orient="records"),
    }

print("Val backtest (calibrated)...", file=sys.stderr, flush=True)
vt = backtest(val, long_m, short_m, cal_fn, "val", use_calibrated=True)
print("OOS backtest (calibrated)...", file=sys.stderr, flush=True)
ot = backtest(oos, long_m, short_m, cal_fn, "oos", use_calibrated=True)

fi = dict(zip(feat_cols, long_m.feature_importances_.tolist()))
model_b64 = base64.b64encode(gzip.compress(pickle.dumps(long_m, protocol=4))).decode()
cal_b64   = base64.b64encode(gzip.compress(pickle.dumps(cal_lr,  protocol=4))).decode()

print(json.dumps({
    "val":  summarize(vt, "Val"),
    "oos":  summarize(ot, "OOS"),
    "fi":   dict(sorted(fi.items(), key=lambda x:-x[1])[:20]),
    "auc":  {"long": long_auc, "short": short_auc},
    "threshold": {"raw": raw_thresh, "calibrated": cal_thresh, "pct": thresh_pct},
    "model_b64": model_b64,
    "cal_b64":   cal_b64,
}))
'''


def _detect_splits(feat: pd.DataFrame) -> dict:
    """
    Auto-detect train/val/OOS splits based on available data range.
    With 3+ years: 2022-2024 train, 2025 val, 2026 OOS
    With 6 months:  Dec-Jan train, Feb val, Mar+ OOS (same as v3)
    """
    start = feat.index[0]
    end   = feat.index[-1]
    span_days = (end - start).days

    if span_days > 400:  # 3+ years
        print(f"  Full dataset detected ({span_days} days) — using 3-year splits", flush=True)
        return {
            "train_end":  "2024-12-31 23:59",
            "val_start":  "2025-01-01",
            "val_end":    "2025-11-30 23:59",
            "oos_start":  "2025-12-01",
            "label":      "3yr",
        }
    else:
        print(f"  Short dataset ({span_days} days) — using 6-month splits (run with full data for best results)", flush=True)
        return {
            "train_end":  "2026-02-09 23:59",
            "val_start":  "2026-02-01",
            "val_end":    "2026-02-09 23:59",
            "oos_start":  "2026-03-24",
            "label":      "6mo",
        }


def _remote_train(feat, splits, window_label):
    t_end   = pd.Timestamp(splits["train_end"],  tz="UTC")
    v_start = pd.Timestamp(splits["val_start"],  tz="UTC")
    v_end   = pd.Timestamp(splits["val_end"],    tz="UTC")
    o_start = pd.Timestamp(splits["oos_start"],  tz="UTC")

    train = label_bars(feat[feat.index <= t_end].copy())
    val   = label_bars(feat[(feat.index >= v_start) & (feat.index <= v_end)].copy())
    oos   = label_bars(feat[feat.index >= o_start].copy())

    print(f"  [{window_label}] Train {train.index[0].date()} - {train.index[-1].date()} ({len(train)} bars)", flush=True)
    print(f"  [{window_label}] Val   {val.index[0].date()} - {val.index[-1].date()} ({len(val)} bars)", flush=True)
    print(f"  [{window_label}] OOS   {oos.index[0].date()} - {oos.index[-1].date()} ({len(oos)} bars)", flush=True)

    for name, df in [("train", train), ("val", val), ("oos", oos)]:
        ll = df["long_label"]
        valid = ll[ll >= 0]
        if len(valid):
            print(f"  [{window_label}] {name}: {len(valid)} valid, pos_rate={valid.mean():.3f}", flush=True)

    payload = {"train": train, "val": val, "oos": oos, "cfg": CFG, "feat_cols": FEATURE_COLS}
    data_bytes = gzip.compress(pickle.dumps(payload, protocol=4), compresslevel=1)
    print(f"  [{window_label}] Payload {len(data_bytes)/1e6:.1f} MB → remote...", flush=True)

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
    thresh = results.get("threshold", {})
    print(f"\n  [{window_label}] AUC LONG={results['auc']['long']:.4f}  SHORT={results['auc']['short']:.4f}", flush=True)
    print(f"  [{window_label}] Threshold: raw={thresh.get('raw',0):.4f}  "
          f"calibrated={thresh.get('calibrated',0):.4f}  (p{thresh.get('pct',75)})", flush=True)
    for sk, label in [("val","VALIDATION"), ("oos","OOS")]:
        m = results[sk]
        print(f"\n{'='*60}")
        print(f"  [{window_label}] {label}: {m.get('n_trades',0)} trades, {m.get('n_days',0)} days")
        print(f"{'='*60}")
        if not m.get("n_trades"):
            print("  No trades."); continue
        print(f"  Total PnL     : ${m['total_pnl']:>10,.0f}")
        print(f"  Weekly avg    : ${m['weekly_avg']:>10,.0f}")
        print(f"  Win rate      : {m['win_rate']*100:>9.1f}%")
        print(f"  Trades/day    : {m['trades_per_day']:>10.1f}")
        print(f"  Max drawdown  : ${m['max_drawdown']:>10,.0f}")
        print(f"  Sharpe        : {m['sharpe_daily']:>10.2f}")
        for r, d in m.get("by_exit", {}).items():
            print(f"    Exit {r}: n={d['n']}, PnL=${d['pnl']:,.0f}")
    top_fi = list(results.get("fi", {}).keys())[:10]
    print(f"\n  [{window_label}] Top features: {top_fi}", flush=True)


def main():
    print("=== ML Scalper v5: Class-Balanced + Calibrated ===", flush=True)
    print(f"scale_pos_weight={CFG['scale_pos_weight']}  "
          f"pt_atr={CFG['pt_atr']}  sl_atr={CFG['sl_atr']}", flush=True)

    print("\n[1] Loading microstructure features...", flush=True)
    micro = pd.read_parquet(DATA / "mnq_microstructure_5min.parquet")
    micro.index = pd.to_datetime(micro.index, utc=True)
    print(f"  {len(micro)} bars, {micro.index[0].date()} to {micro.index[-1].date()}", flush=True)

    print("[2] Building features...", flush=True)
    feat = build_features(micro)

    splits = _detect_splits(feat)

    print(f"\n[3] Training walk-forward window ({splits['label']})...", flush=True)
    r1 = _remote_train(feat, splits, "W1")

    MODELS = BASE / "ml_intraday_v3" / "models"
    MODELS.mkdir(exist_ok=True)

    if "model_b64" in r1:
        model_bytes = gzip.decompress(base64.b64decode(r1["model_b64"]))
        with open(MODELS / "ml_scalper_v5_long.pkl", "wb") as f:
            f.write(model_bytes)
        print(f"Model saved: ml_scalper_v5_long.pkl ({len(model_bytes)/1e3:.0f} KB)", flush=True)

    if "cal_b64" in r1:
        cal_bytes = gzip.decompress(base64.b64decode(r1["cal_b64"]))
        with open(MODELS / "ml_scalper_v5_cal.pkl", "wb") as f:
            f.write(cal_bytes)
        print("Calibrator saved: ml_scalper_v5_cal.pkl", flush=True)

    _print_results(r1, "W1")

    out = {
        "version": "v5", "config": CFG, "features": FEATURE_COLS,
        "splits": splits, "W1": r1,
    }
    out_path = RESULTS / "ml_scalper_v5_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults: {out_path}", flush=True)


if __name__ == "__main__":
    main()
