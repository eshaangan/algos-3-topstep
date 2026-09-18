"""Walk-forward session-direction models. The question is ONE number: out-of-sample
directional accuracy. The account needs >52.5% to beat always-long."""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "/private/tmp/claude-501/-Users-eshaanganguly-Documents-projects-algos-3-topstep/f6967d40-9bb8-470d-b3bc-1b7c827a9ace/scratchpad")
from sessions import build
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

s, f, y = build()
d = pd.concat([f, s[["ret"]], y.rename("y")], axis=1).dropna()
X = d[f.columns].values; R = d["ret"].values; Y = d["y"].values
n = len(d); cols = list(f.columns)
print(f"{n} usable sessions.  base rate up = {(Y>0).mean():.4f}")

TRAIN, STEP = 500, 21          # ~2y train, retrain monthly, expanding-origin
def walkforward(fit_predict, name):
    pred = np.full(n, np.nan)
    for i in range(TRAIN, n, STEP):
        j = min(i+STEP, n)
        pred[i:j] = fit_predict(X[:i], R[:i], Y[:i], X[i:j])
    m = ~np.isnan(pred)
    acc = (np.sign(pred[m]) == Y[m]).mean()
    # dollar terms: always-long comparison on the same sessions
    pnl = (np.sign(pred[m]) * R[m]).mean()
    lng = R[m].mean()
    k = m.sum(); se = np.sqrt(acc*(1-acc)/k)
    print(f"{name:>22}  n={k:>5}  acc {acc:>6.2%} +-{se:6.2%}  "
          f"t vs 50% {(acc-0.5)/se:>+5.2f}  $/sess {pnl:>+7.2f}  (long {lng:>+7.2f})")
    return acc, pred, m

def _sc(Xa, Xb):
    sc = StandardScaler().fit(Xa); return sc.transform(Xa), sc.transform(Xb)

def lin(Xa, Ra, Ya, Xb):
    a, b = _sc(Xa, Xb); return LinearRegression().fit(a, Ra).predict(b)
def ridge(Xa, Ra, Ya, Xb):
    a, b = _sc(Xa, Xb); return RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(a, Ra).predict(b)
def logit(Xa, Ra, Ya, Xb):
    a, b = _sc(Xa, Xb)
    return LogisticRegression(C=0.1, max_iter=2000).fit(a, Ya).predict_proba(b)[:, 1] - 0.5
def rf(Xa, Ra, Ya, Xb):
    return RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=40,
        random_state=0, n_jobs=-1).fit(Xa, Ya).predict_proba(Xb)[:, 1] - 0.5
def gb(Xa, Ra, Ya, Xb):
    return GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.03,
        subsample=0.8, random_state=0).fit(Xa, Ya).predict_proba(Xb)[:, 1] - 0.5
def always_long(Xa, Ra, Ya, Xb): return np.ones(len(Xb))
def fade_prev(Xa, Ra, Ya, Xb):   return -np.sign(Xb[:, cols.index("r1")])
def follow_prev(Xa, Ra, Ya, Xb): return  np.sign(Xb[:, cols.index("r1")])
def ma200(Xa, Ra, Ya, Xb):       return  np.sign(Xb[:, cols.index("ma200")])

print("\n--- benchmarks -------------------------------------------------------")
for fn, nm in ((always_long, "always long"), (follow_prev, "follow prev session"),
               (fade_prev, "FADE prev session"), (ma200, "above MA200")):
    walkforward(fn, nm)
print("\n--- learned models ---------------------------------------------------")
out = {}
for fn, nm in ((lin, "linear regression"), (ridge, "ridge"), (logit, "logistic"),
               (rf, "random forest"), (gb, "gradient boosting")):
    out[nm] = walkforward(fn, nm)
print(f"\nthreshold to beat always-long: 52.5% directional accuracy")
