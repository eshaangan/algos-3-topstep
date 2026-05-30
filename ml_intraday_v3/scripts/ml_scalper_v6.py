"""
ML Scalper v6 — time-of-day segmented models.

Trains 3 separate LightGBM models, one per intraday session:
  OPEN    (10:00–11:00 ET): ORB dynamics, highest OFI predictability
  MIDDAY  (11:00–14:00 ET): VWAP mean-reversion, lower volatility
  AFTNOON (14:00–15:00 ET): directional continuation, pre-close flow

Academic basis: "intraday intervals treated as individual time series outperform
methods treating data as one continuous sample" (Duke/Morgan Stanley, 2025).

Each session model has different optimal features and threshold. At inference,
the live runner selects the correct model based on bar timestamp.

Runs after v5 — use the same CFG.pt_atr/sl_atr values from barrier_grid_search.py.
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

# Session definitions (ET hours, UTC = ET + 5)
SESSIONS = {
    "open":     {"h_et_start": 10, "h_et_end": 11, "h_utc_start": 15, "h_utc_end": 16},
    "midday":   {"h_et_start": 11, "h_et_end": 14, "h_utc_start": 16, "h_utc_end": 19},
    "afternoon":{"h_et_start": 14, "h_et_end": 15, "h_utc_start": 19, "h_utc_end": 20},
}

CFG = dict(
    pt_atr=1.5, sl_atr=1.0, horizon_bars=12, atr_period=10,
    n_estimators=600, learning_rate=0.02, num_leaves=31,
    min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.3, reg_lambda=1.5,
    scale_pos_weight=4.5,
    threshold_pct=75,
    long_threshold=None, short_threshold=0.99,
    min_atr_pts=5.0,
    n_contracts=6,
    pt_mult=1.5, sl_mult=1.0,
    max_trades_per_day=4,
    max_daily_loss_pts=100,
    cooldown_bars=2,
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


def filter_session(df: pd.DataFrame, session: str) -> pd.DataFrame:
    s = SESSIONS[session]
    h_utc = df.index.hour
    return df[(h_utc >= s["h_utc_start"]) & (h_utc < s["h_utc_end"])]


REMOTE_SCRIPT = r'''
import sys, gzip, pickle, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

payload   = pickle.loads(gzip.decompress(sys.stdin.buffer.read()))
sessions  = payload["sessions"]   # dict: session_name → {train, val, oos, threshold}
cfg       = payload["cfg"]
feat_cols = payload["feat_cols"]

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

results   = {}
models    = {}

for sess_name, sess_data in sessions.items():
    train = sess_data["train"]
    val   = sess_data["val"]
    oos   = sess_data["oos"]
    print(f"\n[{sess_name}] Training...", file=sys.stderr, flush=True)

    lc   = "long_label"
    mask = (train[lc] >= 0) & train[feat_cols].notna().all(axis=1)
    X    = train.loc[mask, feat_cols].astype(np.float32)
    y    = train.loc[mask, lc].astype(np.int32)
    print(f"  [{sess_name}] n={len(X)} pos_rate={y.mean():.3f}", file=sys.stderr, flush=True)

    if len(X) < 40 or y.sum() < 10:
        print(f"  [{sess_name}] Insufficient samples — skipping", file=sys.stderr, flush=True)
        results[sess_name] = {"skipped": True, "reason": "insufficient_samples"}
        continue

    split = int(len(X) * 0.8)
    model = lgb.LGBMClassifier(
        n_estimators=cfg["n_estimators"], learning_rate=cfg["learning_rate"],
        num_leaves=cfg["num_leaves"], min_child_samples=cfg["min_child_samples"],
        subsample=cfg["subsample"], colsample_bytree=cfg["colsample_bytree"],
        reg_alpha=cfg.get("reg_alpha",0), reg_lambda=cfg.get("reg_lambda",0),
        scale_pos_weight=cfg.get("scale_pos_weight",1.0),
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X.iloc[:split], y.iloc[:split],
              eval_set=[(X.iloc[split:], y.iloc[split:])],
              callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)])
    auc = roc_auc_score(y.iloc[split:], model.predict_proba(X.iloc[split:])[:,1])
    print(f"  [{sess_name}] AUC={auc:.4f}", file=sys.stderr, flush=True)

    # Calibration + dynamic threshold from val
    val_mask = (val["long_label"] >= 0) & val[feat_cols].notna().all(axis=1)
    X_val_all = val[val[feat_cols].notna().all(axis=1)][feat_cols].astype(np.float32)
    p_val = model.predict_proba(X_val_all)[:,1]

    # Platt scaling
    if val_mask.sum() >= 10:
        X_vm = val.loc[val_mask, feat_cols].astype(np.float32)
        y_vm = val.loc[val_mask, "long_label"].astype(np.int32)
        logit = np.log(p_val.clip(1e-6,1-1e-6)/(1-p_val.clip(1e-6,1-1e-6)))
        X_vm_logit = np.log(model.predict_proba(X_vm)[:,1].clip(1e-6,1-1e-6) /
                           (1 - model.predict_proba(X_vm)[:,1].clip(1e-6,1-1e-6)))
        lr = LogisticRegression(C=1e4).fit(X_vm_logit.reshape(-1,1), y_vm)
    else:
        lr = None

    thresh_pct = cfg.get("threshold_pct", 75)
    if lr is not None:
        logit_all = np.log(p_val.clip(1e-6,1-1e-6)/(1-p_val.clip(1e-6,1-1e-6)))
        p_cal = lr.predict_proba(logit_all.reshape(-1,1))[:,1]
        threshold = float(np.percentile(p_cal, thresh_pct))
        use_cal = True
    else:
        threshold = float(np.percentile(p_val, thresh_pct))
        use_cal = False

    print(f"  [{sess_name}] threshold=p{thresh_pct}={threshold:.4f} "
          f"(calibrated={use_cal})", file=sys.stderr, flush=True)

    # OOS backtest for this session
    def backtest_sess(df, threshold, lr=None):
        valid = df[feat_cols].notna().all(axis=1)
        df = df[valid].copy()
        if df.empty: return pd.DataFrame()
        X = df[feat_cols].astype(np.float32)
        p_raw = model.predict_proba(X)[:,1]
        if lr is not None:
            logit = np.log(p_raw.clip(1e-6,1-1e-6)/(1-p_raw.clip(1e-6,1-1e-6)))
            p = lr.predict_proba(logit.reshape(-1,1))[:,1]
        else:
            p = p_raw
        df["p_long"] = p
        trades = []
        in_trade = False; cooldown = 0
        ep = ea = 0.0; ed = eb = 0
        c = df["close"].values; h = df["high"].values; l = df["low"].values
        av = df["atr"].values; pl = df["p_long"].values; idx = df.index
        nc=cfg["n_contracts"]; ptm=cfg["pt_mult"]; slm=cfg["sl_mult"]
        mdd=cfg["max_daily_loss_pts"]; mina=cfg["min_atr_pts"]
        hor=cfg["horizon_bars"]; cd=cfg["cooldown_bars"]
        scale=2.0*nc; dp={}; dtc={}
        mtd = max(1, cfg["max_trades_per_day"] // 3)  # per-session limit
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
                               "direction":"LONG","pnl_pts":round(pts,2),
                               "pnl_dollars":round(pts*scale,2),"exit_reason":why,"session":sess_name})
                in_trade=False; cooldown=cd; continue
            if cooldown>0: cooldown-=1; continue
            if dp[ds]<=-mdd or dtc[ds]>=mtd: continue
            if np.isnan(av[i]) or av[i]<mina: continue
            if pl[i]>=threshold:
                in_trade=True;ed=1;ep=c[i];ea=av[i];eb=i;dtc[ds]+=1
        return pd.DataFrame(trades)

    oos_trades = backtest_sess(oos, threshold, lr)
    n = len(oos_trades)
    wr = float((oos_trades["pnl_dollars"]>0).mean()) if n > 0 else 0
    pnl = float(oos_trades["pnl_dollars"].sum()) if n > 0 else 0
    print(f"  [{sess_name}] OOS: {n} trades, WR={wr:.1%}, PnL=${pnl:,.0f}",
          file=sys.stderr, flush=True)

    fi = dict(zip(feat_cols, model.feature_importances_.tolist()))
    model_b64 = base64.b64encode(gzip.compress(pickle.dumps(model, protocol=4))).decode()
    cal_b64   = base64.b64encode(gzip.compress(pickle.dumps(lr,    protocol=4))).decode() if lr else None

    results[sess_name] = {
        "auc": auc, "threshold": threshold, "calibrated": use_cal,
        "n_oos_trades": n, "oos_wr": wr, "oos_pnl": pnl,
        "fi": dict(sorted(fi.items(), key=lambda x:-x[1])[:10]),
        "model_b64": model_b64, "cal_b64": cal_b64,
        "oos_trades": oos_trades.to_dict(orient="records") if n > 0 else [],
    }

import base64
print(json.dumps(results))
'''


def _detect_splits(feat):
    span = (feat.index[-1] - feat.index[0]).days
    if span > 400:
        return {"train_end": "2024-12-31 23:59", "val_start": "2025-01-01",
                "val_end": "2025-11-30 23:59", "oos_start": "2025-12-01", "label": "3yr"}
    return {"train_end": "2026-02-09 23:59", "val_start": "2026-02-01",
            "val_end": "2026-02-09 23:59", "oos_start": "2026-03-24", "label": "6mo"}


def main():
    print("=== ML Scalper v6: Time-of-Day Segmented Models ===", flush=True)

    micro = pd.read_parquet(DATA / "mnq_microstructure_5min.parquet")
    micro.index = pd.to_datetime(micro.index, utc=True)
    print(f"Dataset: {len(micro)} bars, {micro.index[0].date()} to {micro.index[-1].date()}", flush=True)

    feat   = build_features(micro)
    splits = _detect_splits(feat)
    t_end  = pd.Timestamp(splits["train_end"], tz="UTC")
    v_s    = pd.Timestamp(splits["val_start"],  tz="UTC")
    v_e    = pd.Timestamp(splits["val_end"],    tz="UTC")
    o_s    = pd.Timestamp(splits["oos_start"],  tz="UTC")

    train_all = label_bars(feat[feat.index <= t_end].copy())
    val_all   = label_bars(feat[(feat.index >= v_s) & (feat.index <= v_e)].copy())
    oos_all   = label_bars(feat[feat.index >= o_s].copy())

    sessions_payload = {}
    for sess_name in SESSIONS:
        tr = filter_session(train_all, sess_name)
        vl = filter_session(val_all,   sess_name)
        oo = filter_session(oos_all,   sess_name)
        pos = (tr["long_label"] == 1).sum() if len(tr) else 0
        print(f"  {sess_name}: train={len(tr)} bars ({pos} positive), val={len(vl)}, oos={len(oo)}", flush=True)
        sessions_payload[sess_name] = {"train": tr, "val": vl, "oos": oo}

    payload    = {"sessions": sessions_payload, "cfg": CFG, "feat_cols": FEATURE_COLS}
    data_bytes = gzip.compress(pickle.dumps(payload, protocol=4), compresslevel=1)
    print(f"\nPayload: {len(data_bytes)/1e6:.1f} MB → remote...", flush=True)

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

    results = json.loads(stdout_b.decode().strip().splitlines()[-1])

    MODELS = BASE / "ml_intraday_v3" / "models"
    MODELS.mkdir(exist_ok=True)

    print("\n=== RESULTS ===")
    total_trades = 0
    combined_pnl = 0
    for sess_name, r in results.items():
        if r.get("skipped"):
            print(f"  {sess_name}: SKIPPED ({r.get('reason')})"); continue
        print(f"  {sess_name}: AUC={r['auc']:.4f}  thresh={r['threshold']:.4f}  "
              f"OOS {r['n_oos_trades']} trades  WR={r['oos_wr']:.1%}  PnL=${r['oos_pnl']:,.0f}")
        print(f"    Top features: {list(r['fi'].keys())[:5]}")
        total_trades += r["n_oos_trades"]
        combined_pnl += r["oos_pnl"]

        if r.get("model_b64"):
            mb = gzip.decompress(base64.b64decode(r["model_b64"]))
            with open(MODELS / f"ml_scalper_v6_{sess_name}.pkl", "wb") as f:
                f.write(mb)
        if r.get("cal_b64"):
            cb = gzip.decompress(base64.b64decode(r["cal_b64"]))
            with open(MODELS / f"ml_scalper_v6_{sess_name}_cal.pkl", "wb") as f:
                f.write(cb)

    print(f"\n  Combined OOS: {total_trades} trades, ${combined_pnl:,.0f} PnL")

    out_path = RESULTS / "ml_scalper_v6_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "splits": splits, "sessions": list(SESSIONS.keys())},
                  f, indent=2, default=str)
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()
