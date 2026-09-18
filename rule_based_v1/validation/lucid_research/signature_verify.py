"""Scrutinise the signature result before believing it.

The first run beat always-long in dollars ($21.41 vs $14.10) -- the first of nine
model families to do so. It also returned NaN from the leakage audit, which means
the audit did NOT run. Two leakage bugs have already been found this session, so
this checks four things before the result counts:

  1. why the audit was NaN, and run it properly
  2. dev/val split -- one era is not a result
  3. a PAIRED test on the per-session dollar difference vs always-long
  4. whether the edge is direction or just implicit position sizing
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

src = open("rule_based_v1/validation/lucid_research/signature_features.py").read()
ns = {"np": np}; exec(src[src.index("def sig(path"):src.index("rows, idx = [], []")], ns)
sig = ns["sig"]

ET = "America/New_York"
b = pd.read_parquet("data/processed/mnq_1m_all.parquet").tz_convert(ET).sort_index()
b = b[~b.index.duplicated(keep="last")]
mm = b.index.hour*60 + b.index.minute
b = b.assign(sid=(mm >= 18*60).cumsum(), mins=mm)
rows, idx = [], []
for _, g in b.groupby("sid"):
    g = g[(g["mins"] <= 16*60+45) | (g["mins"] >= 18*60)]
    if len(g) < 200: continue
    p = g["close"].values.astype(float); v = g["vol"].values.astype(float)
    s = p.std()
    if s <= 0: continue
    t = np.linspace(0, 1, len(p)); P = (p - p[0])/s
    V = np.cumsum(v)/max(v.sum(), 1.0)
    rows.append(np.concatenate([sig(np.column_stack([t, P]), 3),
                                sig(np.column_stack([P, V]), 2)]))
    idx.append((g.index[0], (g["close"].iloc[-1]-g["open"].iloc[0])*2.0))

S = pd.DataFrame(rows, index=pd.DatetimeIndex([i for i, _ in idx]))
S.columns = [f"sig{i}" for i in range(S.shape[1])]
R_all = pd.Series([r for _, r in idx], index=S.index)

# ---- 1. why NaN? ----------------------------------------------------------
const = [c for c in S.columns if S[c].std() == 0 or not np.isfinite(S[c]).all()]
print(f"{S.shape[1]} terms; {len(const)} are constant or non-finite -> {const}")
S = S.drop(columns=const)
F = S.shift(1)
d = pd.concat([F, R_all.rename("ret")], axis=1).dropna()
X = d[S.columns].values; R = d["ret"].values; Y = np.where(R > 0, 1, -1); n = len(d)
tt = [abs(np.corrcoef(X[:, j], R)[0, 1])*np.sqrt((n-2)/max(1e-12, 1-np.corrcoef(X[:, j], R)[0, 1]**2))
      for j in range(X.shape[1])]
print(f"leakage audit NOW RUNS: max |t| = {max(tt):.2f}  ({'FLAG' if max(tt) > 4 else 'ok'})")

# ---- 2-4. walk-forward with era split and a paired test -------------------
TRAIN, STEP = 500, 21
pred = np.full(n, np.nan)
for i in range(TRAIN, n, STEP):
    j = min(i+STEP, n); sc = StandardScaler().fit(X[:i])
    m = LogisticRegression(C=0.05, max_iter=3000).fit(sc.transform(X[:i]), Y[:i])
    pred[i:j] = m.predict_proba(sc.transform(X[i:j]))[:, 1] - 0.5
msk = ~np.isnan(pred)
sgn = np.sign(pred[msk]); r = R[msk]; k = len(r)
H = k//2
print(f"\n{'era':>5} {'n':>5} {'acc':>7} {'model $/s':>10} {'long $/s':>9} "
      f"{'diff':>7} {'paired t':>9}")
for tag, a, bnd in (("dev", 0, H), ("val", H, k), ("all", 0, k)):
    s_, r_ = sgn[a:bnd], r[a:bnd]
    diff = s_*r_ - r_                        # paired per-session difference
    t = diff.mean()/(diff.std(ddof=1)/np.sqrt(len(diff))) if diff.std() > 0 else 0
    print(f"{tag:>5} {len(r_):>5} {(s_ == np.sign(r_)).mean():>7.2%} "
          f"{(s_*r_).mean():>10,.2f} {r_.mean():>9,.2f} {diff.mean():>7,.2f} {t:>9.2f}")
print(f"\nshare of sessions the model goes SHORT: {(sgn < 0).mean():.1%}")
