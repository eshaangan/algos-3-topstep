"""Classic technical analysis, tested head-to-head on session direction.

The claim being checked: RSI/MACD/Bollinger/Stochastic/ATR are deterministic
functions of price history already spanned by the momentum+volatility+MA feature
set, so they should add nothing. Measured rather than asserted.
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "rule_based_v1/validation/lucid_research")
from sessions import build
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

s, f, y = build()
c = s["close"]; h = s["high"]; l = s["low"]; o = s["open"]

def rsi(x, k):
    d = x.diff(); up = d.clip(lower=0).rolling(k).mean()
    dn = (-d.clip(upper=0)).rolling(k).mean()
    return 100 - 100/(1 + up/dn.replace(0, np.nan))

ta = pd.DataFrame(index=s.index)
for k in (2, 7, 14):
    ta[f"rsi{k}"] = rsi(c, k).shift(1)                      # lagged to the open
e12 = c.ewm(span=12).mean(); e26 = c.ewm(span=26).mean()
macd = e12 - e26
ta["macd"] = macd.shift(1); ta["macd_sig"] = (macd - macd.ewm(span=9).mean()).shift(1)
m20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
ta["bb_pctb"] = ((c - (m20 - 2*sd20)) / (4*sd20)).shift(1)  # Bollinger %B
ta["bb_width"] = (4*sd20/m20).shift(1)
lo14 = l.rolling(14).min(); hi14 = h.rolling(14).max()
ta["stoch"] = ((c - lo14)/(hi14 - lo14)).shift(1)           # stochastic %K
tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
ta["atr14"] = tr.rolling(14).mean().shift(1)
ta["atr_ratio"] = (tr.rolling(5).mean()/tr.rolling(20).mean()).shift(1)
up = h.diff(); dn = -l.diff()
pdm = np.where((up > dn) & (up > 0), up, 0.0); ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
atr = tr.rolling(14).mean()
pdi = 100*pd.Series(pdm, index=s.index).rolling(14).mean()/atr
ndi = 100*pd.Series(ndm, index=s.index).rolling(14).mean()/atr
ta["adx"] = (100*(pdi-ndi).abs()/(pdi+ndi)).rolling(14).mean().shift(1)
ta["cci"] = (((c-m20)/(0.015*c.rolling(20).std()))).shift(1)
ta["obv"] = (np.sign(c.diff())*s["vol"]).cumsum().pct_change(5).shift(1)

SETS = {"baseline (momentum/vol/MA)": f,
        "classic TA only": ta,
        "baseline + classic TA": pd.concat([f, ta], axis=1)}
TRAIN, STEP = 500, 21
for name, F in SETS.items():
    d = pd.concat([F, s[["ret"]], y.rename("y")], axis=1).dropna()
    X = d[F.columns].values; R = d["ret"].values; Y = d["y"].values; n = len(d)
    print(f"\n{name}  ({F.shape[1]} features, {n} rows)")
    for mk, lab in ((lambda: "logit", "logistic"), (lambda: "gb", "gradient boost")):
        pred = np.full(n, np.nan)
        for i in range(TRAIN, n, STEP):
            j = min(i+STEP, n); sc = StandardScaler().fit(X[:i])
            a, b = sc.transform(X[:i]), sc.transform(X[i:j])
            if lab == "logistic":
                m = LogisticRegression(C=0.1, max_iter=2000).fit(a, Y[:i])
            else:
                m = GradientBoostingClassifier(n_estimators=150, max_depth=2,
                    learning_rate=0.03, subsample=0.8, random_state=0).fit(a, Y[:i])
            pred[i:j] = m.predict_proba(b)[:, 1] - 0.5
        msk = ~np.isnan(pred); k = msk.sum()
        acc = (np.sign(pred[msk]) == Y[msk]).mean(); se = np.sqrt(acc*(1-acc)/k)
        pnl = (np.sign(pred[msk])*R[msk]).mean()
        print(f"  {lab:>16}  acc {acc:>6.2%} +-{se:5.2%}  t {(acc-.5)/se:>+5.2f}  "
              f"$/sess {pnl:>+7.2f}   (always-long on same rows {R[msk].mean():>+7.2f})")
