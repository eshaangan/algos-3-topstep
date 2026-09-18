"""Truncated path signatures as session-direction features. Model family #9.

WHY THIS IS NOT JUST ANOTHER INDICATOR SET. Every feature tested so far -- momentum
at 7 horizons, volatility, MA distance, RSI/MACD/Bollinger/ADX/CCI/OBV -- is a
POINT-IN-TIME summary. None of them can tell "rose then fell" from "fell then rose"
when the endpoints match. The signature is exactly the object that encodes ORDER:
its level-2 terms are the signed areas of the path, which is genuinely information
the previous 36 features could not represent.

WHAT THE INSTAGRAM CLAIM ACTUALLY SAYS. R^2 of 66/89/98% at orders 1/3/11 is for
APPROXIMATING INDICATORS, not for predicting returns. Universal approximation of a
basis says nothing about whether that basis forecasts. This repo already measured
classic indicators at 54.16% accuracy, losing to always-long in dollars.

MATHS (no library needed at low order). For a discrete path X_0..X_T in R^d:
    S^i    = X_T^i - X_0^i
    S^ij   = sum_k  A_k^i * dX_k^j        where A_k^i is the running level-1 term
    S^ijk  = sum_k  A_k^ij * dX_k^k
This is the standard iterated-sum construction; Chen's relation is not needed since
we compute each path from scratch.

PATH USED: the PREVIOUS session's intraday minute bars, time-augmented and
normalised, so everything is strictly lagged. Level 2 antisymmetric part = Levy area.
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "rule_based_v1/validation/lucid_research")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

ET = "America/New_York"
b = pd.read_parquet("data/processed/mnq_1m_all.parquet").tz_convert(ET).sort_index()
b = b[~b.index.duplicated(keep="last")]
mm = b.index.hour*60 + b.index.minute
b = b.assign(sid=(mm >= 18*60).cumsum(), mins=mm)

def sig(path, order=3):
    """Truncated signature of a d-dim path, levels 1..order, as a flat vector.

    Iterated sums: A_1[k] = sum_{m<=k} dX[m], and
                   A_L[k] = sum_{m<=k} A_{L-1}[m-1] (x) dX[m]
    so each level shifts the running lower level by one step before the outer
    product -- that shift is what makes it an iterated INTEGRAL rather than a
    plain product of sums.
    """
    X = np.asarray(path, float)
    dX = np.diff(X, axis=0)                       # (T, d)
    T, d = dX.shape
    prev = np.cumsum(dX, axis=0)                  # level 1, (T, d)
    out = [prev[-1].copy()]
    for level in range(2, order + 1):
        zero = np.zeros((1,) + prev.shape[1:])
        prev_sh = np.concatenate([zero, prev[:-1]], axis=0)     # A_{L-1}[m-1]
        inc = dX.reshape((T,) + (1,) * (level - 1) + (d,))
        prev = np.cumsum(prev_sh[..., None] * inc, axis=0)
        out.append(prev[-1].ravel())
    return np.concatenate(out)


rows, idx = [], []
for _, g in b.groupby("sid"):
    g = g[(g["mins"] <= 16*60+45) | (g["mins"] >= 18*60)]
    if len(g) < 200: continue
    p = g["close"].values.astype(float)
    v = g["vol"].values.astype(float)
    s = p.std()
    if s <= 0: continue
    t = np.linspace(0, 1, len(p))
    P = (p - p[0]) / s                                   # scale-free price path
    V = (np.cumsum(v) / max(v.sum(), 1.0))               # normalised volume clock
    rows.append(np.concatenate([sig(np.column_stack([t, P]), 3),
                                sig(np.column_stack([P, V]), 2)]))
    idx.append((g.index[0], (g["close"].iloc[-1]-g["open"].iloc[0])*2.0))

ts = [i for i, _ in idx]; ret = np.array([r for _, r in idx])
S = pd.DataFrame(rows, index=pd.DatetimeIndex(ts))
S.columns = [f"sig{i}" for i in range(S.shape[1])]
F = S.shift(1)                                           # strictly lagged
d = pd.concat([F, pd.Series(ret, index=S.index, name="ret")], axis=1).dropna()
X = d[F.columns].values; R = d["ret"].values
Y = np.where(R > 0, 1, -1); n = len(d)
print(f"{n} sessions, {X.shape[1]} signature terms (orders 1-3 on (t,price), 1-2 on (price,vol))")

# leakage audit the ledger demands
tt = []
for j in range(X.shape[1]):
    c = np.corrcoef(X[:, j], R)[0, 1]
    tt.append(abs(c)*np.sqrt((n-2)/max(1e-12, 1-c*c)))
print(f"leakage audit: max |t| of any term vs SAME-session return = {max(tt):.2f} "
      f"({'FLAG' if max(tt) > 4 else 'ok'})")

TRAIN, STEP = 500, 21
for lab in ("logistic", "gradient boost"):
    pred = np.full(n, np.nan)
    for i in range(TRAIN, n, STEP):
        j = min(i+STEP, n); sc = StandardScaler().fit(X[:i])
        a, bb = sc.transform(X[:i]), sc.transform(X[i:j])
        m = (LogisticRegression(C=0.05, max_iter=3000) if lab == "logistic"
             else GradientBoostingClassifier(n_estimators=150, max_depth=2,
                  learning_rate=0.03, subsample=0.8, random_state=0))
        pred[i:j] = m.fit(a, Y[:i]).predict_proba(bb)[:, 1] - 0.5
    msk = ~np.isnan(pred); k = msk.sum()
    acc = (np.sign(pred[msk]) == Y[msk]).mean(); se = np.sqrt(acc*(1-acc)/k)
    print(f"  {lab:>15}  n={k}  acc {acc:>6.2%} +-{se:5.2%}  t {(acc-.5)/se:>+5.2f}  "
          f"$/sess {(np.sign(pred[msk])*R[msk]).mean():>+7.2f}  "
          f"(always-long {R[msk].mean():>+7.2f})")
