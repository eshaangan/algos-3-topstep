"""Build the MNQ session table used by every test below.

A 'session' is the LucidFlex tradable day: 18:00 ET open to 16:45 ET flatten.
Entry is at the session open, so EVERY feature must be knowable at that instant.
"""
import numpy as np, pandas as pd
ET = "America/New_York"; PV = 2.0

def build(path="data/processed/mnq_1m_all.parquet"):
    b = pd.read_parquet(path).tz_convert(ET).sort_index()
    b = b[~b.index.duplicated(keep="last")]
    mm = b.index.hour*60 + b.index.minute
    b = b.assign(sid=(mm >= 18*60).cumsum(), mins=mm)
    rows = []
    for sid, g in b.groupby("sid"):
        g = g[(g["mins"] <= 16*60+45) | (g["mins"] >= 18*60)]
        if len(g) < 200: continue
        o = g["open"].iloc[0]; c = g["close"].iloc[-1]
        rows.append(dict(sid=sid, ts=g.index[0], open=o, close=c,
                         high=g["high"].max(), low=g["low"].min(),
                         ret=(c-o)*PV, low_x=(g["low"].min()-o)*PV,
                         high_x=(g["high"].max()-o)*PV, vol=g["vol"].sum()))
    s = pd.DataFrame(rows).set_index("ts")
    # --- features, all strictly lagged to information available at the open ----
    r = s["ret"] / PV                      # points, session open -> close
    rng = (s["high"] - s["low"])
    f = pd.DataFrame(index=s.index)
    for k in (1, 2, 3, 5, 10, 20, 60):
        f[f"r{k}"] = r.shift(1).rolling(k).sum()
    for k in (5, 20, 60):
        f[f"vol{k}"] = rng.shift(1).rolling(k).mean()
    f["volratio"] = f["vol5"] / f["vol20"]
    f["rng1"] = rng.shift(1)
    f["clo_loc"] = ((s["close"] - s["low"]) / rng.replace(0, np.nan)).shift(1)
    f["gap"] = (s["open"] - s["close"].shift(1))          # overnight gap, known at open
    f["gap_n"] = f["gap"] / f["vol20"]
    for k in (20, 50, 200):
        f[f"ma{k}"] = (s["open"] - s["close"].shift(1).rolling(k).mean()) / f["vol20"]
    f["dow"] = s.index.dayofweek
    f["r1_n"] = f["r1"] / f["vol20"]
    f["r5_n"] = f["r5"] / f["vol20"]
    f["r20_n"] = f["r20"] / f["vol20"]
    f["r60_n"] = f["r60"] / f["vol20"]
    y = np.sign(s["ret"]).replace(0, 1)
    return s, f, y

if __name__ == "__main__":
    s, f, y = build()
    print(f"{len(s)} sessions {s.index[0].date()} -> {s.index[-1].date()}")
    print(f"{f.shape[1]} features, {f.dropna().shape[0]} complete rows")
    print(f"base rate up = {(y>0).mean():.4f}")
    # leakage audit the ledger demands: no feature may correlate with the SAME
    # session's return beyond what a real predictor could.
    d = pd.concat([f, s["ret"]], axis=1).dropna()
    print("\ntop |t| of feature vs SAME-session return (flag |t|>4):")
    n = len(d)
    for col in f.columns:
        c = d[col].corr(d["ret"]); t = c*np.sqrt((n-2)/max(1e-12, 1-c*c))
        if abs(t) > 2: print(f"  {col:>10} corr {c:+.4f}  t {t:+.2f}")
