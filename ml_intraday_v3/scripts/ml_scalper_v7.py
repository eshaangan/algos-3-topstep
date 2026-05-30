"""
ML Scalper v7 — SHORT model revival.

With 3+ years of data, attempts to find a SHORT edge that was not detectable
with 6 months (SHORT AUC was ~0.529 — barely above chance on insufficient samples).

Key approach:
  - Separate scale_pos_weight for SHORT class (independently tuned)
  - SHORT model trained independently from LONG model
  - GATE: only deploys SHORT in backtest/live if OOS SHORT WR >= 50% over >= 30 trades
  - If SHORT gate passes → combined LONG+SHORT model can ~2x trade count

If SHORT gate fails → results are reported but model falls back to LONG-only.

Run AFTER getting 3+ years of MotiveWave data.
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

# SHORT base rate is typically lower than LONG (fewer PT hits on shorts in bull trend)
# Compute SHORT pos rate on your dataset and set scale_pos_weight_short accordingly
CFG = dict(
    pt_atr=1.5, sl_atr=1.0, horizon_bars=12, atr_period=10,
    n_estimators=600, learning_rate=0.02, num_leaves=31,
    min_child_samples=40, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.3, reg_lambda=1.5,
    scale_pos_weight=4.5,        # LONG class weight
    scale_pos_weight_short=5.5,  # SHORT class weight (higher: shorts are rarer)
    threshold_pct=75,
    long_threshold=None, short_threshold=None,  # set dynamically
    min_atr_pts=5.0,
    n_contracts=6, pt_mult=1.5, sl_mult=1.0,
    max_trades_per_day=5,   # 3 LONG + 2 SHORT max
    max_daily_loss_pts=100,
    cooldown_bars=2,
    # Gate: SHORT only deployed if these minimums are met on OOS
    short_gate_min_trades=30,
    short_gate_min_wr=0.50,
)

FEATURE_COLS = [
    "ofi_imb", "lg_ofi_imb", "ofi_accel", "lg_sm_diverge",
    "kyles_lambda", "roll_spread", "large_frac",
    "trade_rate_z", "avg_size_z", "max_run_z",
    "ofi_early_n", "ofi_late_n",
    "ret_1", "ret_3", "ret_6", "ret_12",
    "rsi_5", "rsi_14", "ema9_ratio", "ema21_ratio", "ema9_21_cross",
    "norm_range", "norm_body", "vol_z", "range_pos", "vwap_dev",
    "hour_sin", "hour_cos", "dow", "is_open_30", "is_close_30",
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

    for col in ["trade_rate", "avg_size", "max_run"]:
        ma = out[col].rolling(20).mean()
        sd = out[col].rolling(20).std().replace(0, np.nan)
        out[f"{col}_z"] = (out[col] - ma) / sd

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
    out["hour_sin"]    = np.sin(2*np.pi*h_et/24)
    out["hour_cos"]    = np.cos(2*np.pi*h_et/24)
    out["dow"]         = out.index.dayofweek.astype(float)
    mins = (h_et - 9)*60 + out.index.minute - 30
    out["is_open_30"]  = (mins <= 30).astype(float)
    out["is_close_30"] = (mins >= 360).astype(float)
    return out


def label_bars(df: pd.DataFrame) -> pd.DataFrame:
    c   = df["close"].values; hi = df["high"].values; lo = df["low"].values
    atr = df["atr"].values;   n  = len(df)
    hor = CFG["horizon_bars"]; ptm = CFG["pt_atr"]; slm = CFG["sl_atr"]
    ll = np.full(n, -1, dtype=np.int8)
    sl = np.full(n, -1, dtype=np.int8)
    for i in range(n - hor):
        a = atr[i]
        if np.isnan(a) or a <= 0: continue
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
import base64

def make_model(spw):
    return lgb.LGBMClassifier(
        n_estimators=cfg["n_estimators"], learning_rate=cfg["learning_rate"],
        num_leaves=cfg["num_leaves"], min_child_samples=cfg["min_child_samples"],
        subsample=cfg["subsample"], colsample_bytree=cfg["colsample_bytree"],
        reg_alpha=cfg.get("reg_alpha",0), reg_lambda=cfg.get("reg_lambda",0),
        scale_pos_weight=spw, random_state=42, n_jobs=-1, verbose=-1,
    )

def train_direction(df_tr, df_val, direction, spw):
    lc   = f"{direction}_label"
    mask = (df_tr[lc] >= 0) & df_tr[feat_cols].notna().all(axis=1)
    X    = df_tr.loc[mask, feat_cols].astype(np.float32)
    y    = df_tr.loc[mask, lc].astype(np.int32)
    pos_rate = float(y.mean())
    print(f"  [{direction}] n={len(X)} pos_rate={pos_rate:.3f} spw={spw:.1f}",
          file=sys.stderr, flush=True)
    if len(X) < 40 or y.sum() < 10:
        raise ValueError(f"Insufficient {direction} samples")
    split = int(len(X) * 0.8)
    m = make_model(spw)
    m.fit(X.iloc[:split], y.iloc[:split],
          eval_set=[(X.iloc[split:], y.iloc[split:])],
          callbacks=[lgb.early_stopping(60,verbose=False), lgb.log_evaluation(-1)])
    auc = roc_auc_score(y.iloc[split:], m.predict_proba(X.iloc[split:])[:,1])
    print(f"  [{direction}] AUC={auc:.4f}", file=sys.stderr, flush=True)

    # Calibration
    vmask = (df_val[lc] >= 0) & df_val[feat_cols].notna().all(axis=1)
    X_va  = df_val[df_val[feat_cols].notna().all(axis=1)][feat_cols].astype(np.float32)
    p_va  = m.predict_proba(X_va)[:,1]
    lr = None
    if vmask.sum() >= 10:
        X_vm = df_val.loc[vmask, feat_cols].astype(np.float32)
        y_vm = df_val.loc[vmask, lc].astype(np.int32)
        logit = np.log(m.predict_proba(X_vm)[:,1].clip(1e-6,1-1e-6) /
                       (1 - m.predict_proba(X_vm)[:,1].clip(1e-6,1-1e-6)))
        lr = LogisticRegression(C=1e4).fit(logit.reshape(-1,1), y_vm)
        logit_va = np.log(p_va.clip(1e-6,1-1e-6)/(1-p_va.clip(1e-6,1-1e-6)))
        p_va = lr.predict_proba(logit_va.reshape(-1,1))[:,1]

    thresh_pct = cfg.get("threshold_pct", 75)
    threshold = float(np.percentile(p_va, thresh_pct))
    print(f"  [{direction}] threshold=p{thresh_pct}={threshold:.4f}",
          file=sys.stderr, flush=True)
    return m, lr, auc, threshold

print("Training LONG...", file=sys.stderr, flush=True)
long_m, long_cal, long_auc, long_thresh = train_direction(
    train, val, "long", cfg["scale_pos_weight"])

print("Training SHORT...", file=sys.stderr, flush=True)
try:
    short_m, short_cal, short_auc, short_thresh = train_direction(
        train, val, "short", cfg.get("scale_pos_weight_short", 5.5))
    short_trained = True
except ValueError as e:
    print(f"  SHORT training failed: {e}", file=sys.stderr, flush=True)
    short_trained = False
    short_auc = 0.5
    short_thresh = 0.99

def predict(m, cal, X):
    p = m.predict_proba(X)[:,1]
    if cal is not None:
        logit = np.log(p.clip(1e-6,1-1e-6)/(1-p.clip(1e-6,1-1e-6)))
        p = cal.predict_proba(logit.reshape(-1,1))[:,1]
    return p

def backtest(df, enable_short=True):
    valid = df[feat_cols].notna().all(axis=1)
    df = df[valid].copy()
    if df.empty: return pd.DataFrame()
    X = df[feat_cols].astype(np.float32)
    df["p_long"]  = predict(long_m, long_cal, X)
    df["p_short"] = predict(short_m, short_cal, X) if short_trained else 0.0

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
                           "exit_reason":why})
            in_trade=False; cooldown=cd; continue
        if cooldown>0: cooldown-=1; continue
        if dp[ds]<=-mdd or dtc[ds]>=mtd: continue
        if np.isnan(av[i]) or av[i]<mina: continue
        h_et = (idx[i].hour - 5) % 24
        if h_et < 11 or h_et == 13 or idx[i].weekday()==3: continue
        gl = pl[i] >= long_thresh
        gs = short_trained and enable_short and ps[i] >= short_thresh
        if gl and (not gs or pl[i] >= ps[i]):
            in_trade=True;ed=1;ep=c[i];ea=av[i];eb=i;dtc[ds]+=1
        elif gs:
            in_trade=True;ed=-1;ep=c[i];ea=av[i];eb=i;dtc[ds]+=1
    return pd.DataFrame(trades)

def summarize(tdf, label):
    if tdf is None or tdf.empty: return {"label":label,"n_trades":0}
    wins = tdf[tdf["pnl_dollars"]>0]; losses = tdf[tdf["pnl_dollars"]<=0]
    tdf["date"] = pd.to_datetime(tdf["entry_time"]).dt.strftime("%Y-%m-%d")
    daily = tdf.groupby("date")["pnl_dollars"].sum()
    cum = tdf["pnl_dollars"].cumsum(); max_dd = (cum-cum.cummax()).min()
    n = len(tdf); nd = tdf["date"].nunique()
    by_dir = {}
    for s in ["LONG","SHORT"]:
        sub = tdf[tdf["direction"]==s]
        if len(sub): by_dir[s] = {"n":len(sub),"wr":round((sub["pnl_dollars"]>0).mean(),3),
                                   "pnl":round(float(sub["pnl_dollars"].sum()),2)}
    return {
        "label":label, "n_trades":n, "n_days":int(nd),
        "trades_per_day":round(n/max(nd,1),2),
        "total_pnl":round(float(tdf["pnl_dollars"].sum()),2),
        "win_rate":round(len(wins)/n,4),
        "avg_win":round(float(wins["pnl_dollars"].mean()),2) if len(wins) else 0,
        "avg_loss":round(float(losses["pnl_dollars"].mean()),2) if len(losses) else 0,
        "max_drawdown":round(float(max_dd),2),
        "sharpe_daily":round(float(daily.mean()/daily.std()*(252**0.5)),3) if daily.std()>0 else 0,
        "by_direction": by_dir,
        "trades": tdf.to_dict(orient="records"),
    }

# Check SHORT gate
print("\n--- SHORT Gate Check ---", file=sys.stderr, flush=True)
if short_trained:
    oos_long_only = backtest(oos, enable_short=False)
    oos_combined  = backtest(oos, enable_short=True)
    short_trades = oos_combined[oos_combined["direction"]=="SHORT"] if len(oos_combined) else pd.DataFrame()
    n_short = len(short_trades)
    wr_short = float((short_trades["pnl_dollars"]>0).mean()) if n_short > 0 else 0
    print(f"  SHORT OOS: {n_short} trades, WR={wr_short:.1%}", file=sys.stderr, flush=True)

    gate_passed = (n_short >= cfg["short_gate_min_trades"] and
                   wr_short >= cfg["short_gate_min_wr"])
    print(f"  SHORT gate {'PASSED' if gate_passed else 'FAILED'} "
          f"(need >={cfg['short_gate_min_trades']} trades at >={cfg['short_gate_min_wr']:.0%} WR)",
          file=sys.stderr, flush=True)
else:
    oos_long_only = backtest(oos, enable_short=False)
    oos_combined  = oos_long_only.copy() if len(oos_long_only) else pd.DataFrame()
    gate_passed = False
    n_short = 0; wr_short = 0

fi_long  = dict(zip(feat_cols, long_m.feature_importances_.tolist()))
fi_short = dict(zip(feat_cols, short_m.feature_importances_.tolist())) if short_trained else {}

lm_b64  = base64.b64encode(gzip.compress(pickle.dumps(long_m,  protocol=4))).decode()
sm_b64  = base64.b64encode(gzip.compress(pickle.dumps(short_m, protocol=4))).decode() if short_trained else None
lc_b64  = base64.b64encode(gzip.compress(pickle.dumps(long_cal, protocol=4))).decode() if long_cal else None
sc_b64  = base64.b64encode(gzip.compress(pickle.dumps(short_cal,protocol=4))).decode() if (short_trained and short_cal) else None

print(json.dumps({
    "long_only_oos":  summarize(oos_long_only,  "OOS LONG-only"),
    "combined_oos":   summarize(oos_combined,   "OOS LONG+SHORT"),
    "auc": {"long": long_auc, "short": short_auc},
    "thresholds": {"long": long_thresh, "short": short_thresh if short_trained else None},
    "short_gate": {"passed": gate_passed, "n_trades": n_short, "win_rate": wr_short},
    "fi_long":  dict(sorted(fi_long.items(),  key=lambda x:-x[1])[:15]),
    "fi_short": dict(sorted(fi_short.items(), key=lambda x:-x[1])[:15]),
    "long_model_b64":  lm_b64, "short_model_b64": sm_b64,
    "long_cal_b64":    lc_b64, "short_cal_b64":   sc_b64,
}))
'''


def _detect_splits(feat):
    span = (feat.index[-1] - feat.index[0]).days
    if span > 400:
        print("  Full 3-year dataset — SHORT has enough samples for reliable AUC estimate", flush=True)
        return {"train_end": "2024-12-31 23:59", "val_start": "2025-01-01",
                "val_end": "2025-11-30 23:59", "oos_start": "2025-12-01", "label": "3yr"}
    print("  WARNING: Only 6-month dataset — SHORT AUC likely unreliable (need 3+ years)", flush=True)
    return {"train_end": "2026-02-09 23:59", "val_start": "2026-02-01",
            "val_end": "2026-02-09 23:59", "oos_start": "2026-03-24", "label": "6mo"}


def main():
    print("=== ML Scalper v7: SHORT Revival ===", flush=True)

    micro = pd.read_parquet(DATA / "mnq_microstructure_5min.parquet")
    micro.index = pd.to_datetime(micro.index, utc=True)
    print(f"Dataset: {len(micro)} bars, {micro.index[0].date()} to {micro.index[-1].date()}", flush=True)

    feat   = build_features(micro)
    splits = _detect_splits(feat)
    t_end  = pd.Timestamp(splits["train_end"], tz="UTC")
    v_s    = pd.Timestamp(splits["val_start"],  tz="UTC")
    v_e    = pd.Timestamp(splits["val_end"],    tz="UTC")
    o_s    = pd.Timestamp(splits["oos_start"],  tz="UTC")

    train = label_bars(feat[feat.index <= t_end].copy())
    val   = label_bars(feat[(feat.index >= v_s) & (feat.index <= v_e)].copy())
    oos   = label_bars(feat[feat.index >= o_s].copy())

    for name, df in [("train",train),("val",val),("oos",oos)]:
        ll = df["long_label"]; sl = df["short_label"]
        print(f"  {name}: {len(df)} bars  LONG pos={( ll>=0).sum()} ({ll[ll>=0].mean():.1%})  "
              f"SHORT pos={(sl>=0).sum()} ({sl[sl>=0].mean():.1%})", flush=True)

    payload    = {"train": train, "val": val, "oos": oos, "cfg": CFG, "feat_cols": FEATURE_COLS}
    data_bytes = gzip.compress(pickle.dumps(payload, protocol=4), compresslevel=1)
    print(f"\nPayload {len(data_bytes)/1e6:.1f} MB → remote...", flush=True)

    encoded = base64.b64encode(REMOTE_SCRIPT.encode()).decode()
    proc    = subprocess.Popen(
        ["ssh", SSH_HOST, f"python3 -c \"import base64,sys; exec(base64.b64decode('{encoded}').decode())\""],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    stdout_b, stderr_b = proc.communicate(input=data_bytes)
    for line in stderr_b.decode().splitlines():
        print(f"  [remote] {line}", flush=True)
    if proc.returncode != 0:
        print("Remote failed:", stderr_b.decode()); sys.exit(1)

    r = json.loads(stdout_b.decode().strip().splitlines()[-1])

    print(f"\n=== RESULTS ===")
    print(f"  AUC: LONG={r['auc']['long']:.4f}  SHORT={r['auc']['short']:.4f}")
    print(f"  SHORT gate: {'PASSED ✓' if r['short_gate']['passed'] else 'FAILED ✗'}  "
          f"({r['short_gate']['n_trades']} trades, WR={r['short_gate']['win_rate']:.1%})")

    for key, label in [("long_only_oos","LONG-only"), ("combined_oos","LONG+SHORT")]:
        m = r[key]
        if not m.get("n_trades"): continue
        print(f"\n  [{label}] {m['n_trades']} trades, WR={m['win_rate']:.1%}, "
              f"PnL=${m['total_pnl']:,.0f}, Sharpe={m['sharpe_daily']:.2f}, DD=${m['max_drawdown']:,.0f}")
        for s, d in m.get("by_direction", {}).items():
            print(f"    {s}: n={d['n']}, WR={d['wr']:.1%}, PnL=${d['pnl']:,.0f}")

    if r["short_gate"]["passed"]:
        print("\n  → SHORT gate PASSED — deploy combined LONG+SHORT model")
    else:
        print("\n  → SHORT gate FAILED — deploy LONG-only, monitor SHORT AUC as data grows")

    MODELS = BASE / "ml_intraday_v3" / "models"
    MODELS.mkdir(exist_ok=True)
    for key, fname in [("long_model_b64","ml_scalper_v7_long.pkl"),
                       ("short_model_b64","ml_scalper_v7_short.pkl"),
                       ("long_cal_b64","ml_scalper_v7_long_cal.pkl"),
                       ("short_cal_b64","ml_scalper_v7_short_cal.pkl")]:
        if r.get(key):
            b = gzip.decompress(base64.b64decode(r[key]))
            with open(MODELS / fname, "wb") as f:
                f.write(b)
            print(f"  Saved: {fname}")

    out_path = RESULTS / "ml_scalper_v7_results.json"
    with open(out_path, "w") as f:
        # Don't serialize model bytes into results file
        r_clean = {k: v for k, v in r.items() if not k.endswith("_b64")}
        json.dump({"results": r_clean, "splits": splits, "cfg": CFG},
                  f, indent=2, default=str)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
