"""
Alpha/Beta Attribution Screen ("is it just directional exposure?")

Motivated by Quant Guild lecture 96, "I Bet You've Never Found Alpha (and I Can
Prove It)": regress a strategy's realized P/L on the contemporaneous move in the
instrument it trades. If the intercept (alpha) is statistically indistinguishable
from zero, the strategy is not an edge -- it is net directional exposure plus
transaction costs ("beta in a trench coat").

    Spec 1 (directional):   pnl_t = a + b * mkt_t + e_t
    Spec 2 (+ convexity):   pnl_t = a + b * mkt_t + g * |mkt_t| + e_t

Spec 2 matters for breakout-style strategies (e.g. ORB): a strategy that is paid
whenever the tape moves, regardless of sign, loads on |mkt_t|. Without that term
its volatility harvesting is misattributed to alpha. A strategy whose alpha
survives both specs has return unexplained by direction OR by realized range.

Standard errors are Newey-West (HAC) because daily strategy P/L is serially
correlated (position carryover, regime clustering). Using OLS errors here would
overstate t-stats, which is the exact failure mode this screen exists to catch.

No statsmodels dependency: OLS and the HAC sandwich are computed with numpy so
this runs anywhere the rest of the pipeline runs.

References
----------
Quant Guild, "I Bet You've Never Found Alpha (and I Can Prove It)" (2026).
Newey, W. and West, K. (1987), "A Simple, Positive Semi-Definite,
    Heteroskedasticity and Autocorrelation Consistent Covariance Matrix".
Newey, W. and West, K. (1994), "Automatic Lag Selection in Covariance
    Matrix Estimation".
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------------
# OLS with Newey-West (HAC) standard errors
# ----------------------------------------------------------------------------

def newey_west_lag(n_obs: int) -> int:
    """Automatic bandwidth from Newey-West (1994): floor(4 * (T/100)^(2/9))."""
    if n_obs <= 1:
        return 0
    return int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))


@dataclass
class OLSResult:
    """OLS fit with HAC-robust inference."""
    names: list
    coef: np.ndarray
    se: np.ndarray
    tstat: np.ndarray
    pvalue: np.ndarray
    n_obs: int
    r_squared: float
    hac_lags: int

    def get(self, name: str) -> Dict[str, float]:
        i = self.names.index(name)
        return {
            "coef": float(self.coef[i]),
            "se": float(self.se[i]),
            "t": float(self.tstat[i]),
            "p": float(self.pvalue[i]),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_obs": self.n_obs,
            "r_squared": self.r_squared,
            "hac_lags": self.hac_lags,
            "terms": {n: self.get(n) for n in self.names},
        }


def ols_hac(
    y: np.ndarray,
    X: np.ndarray,
    names: list,
    hac_lags: Optional[int] = None,
) -> OLSResult:
    """
    Ordinary least squares with Newey-West HAC covariance.

    Parameters
    ----------
    y : (T,) response.
    X : (T, k) design matrix; caller supplies the intercept column.
    names : length-k list of term names matching X's columns.
    hac_lags : truncation lag. None selects the Newey-West (1994) rule.

    Returns
    -------
    OLSResult
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    if y.shape[0] != X.shape[0]:
        raise ValueError("y and X must have the same number of rows")
    if len(names) != X.shape[1]:
        raise ValueError("names must match the number of columns in X")

    n, k = X.shape
    if n <= k:
        raise ValueError(f"need more observations ({n}) than parameters ({k})")

    xtx = X.T @ X
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta

    if hac_lags is None:
        hac_lags = newey_west_lag(n)
    hac_lags = int(max(0, min(hac_lags, n - 1)))

    # Sandwich meat: S = Gamma_0 + sum_l w_l (Gamma_l + Gamma_l')
    s = (X * resid[:, None]).T @ (X * resid[:, None])
    for lag in range(1, hac_lags + 1):
        w = 1.0 - lag / (hac_lags + 1.0)
        a = (X[lag:] * resid[lag:, None])
        b = (X[:-lag] * resid[:-lag, None])
        gamma = a.T @ b
        s += w * (gamma + gamma.T)

    cov = xtx_inv @ s @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, beta / se, 0.0)
    dof = max(n - k, 1)
    pvalue = 2.0 * stats.t.sf(np.abs(tstat), df=dof)

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return OLSResult(
        names=list(names),
        coef=beta,
        se=se,
        tstat=np.asarray(tstat, dtype=float),
        pvalue=np.asarray(pvalue, dtype=float),
        n_obs=n,
        r_squared=r_squared,
        hac_lags=hac_lags,
    )


# ----------------------------------------------------------------------------
# Aggregation helpers
# ----------------------------------------------------------------------------

def daily_pnl_from_trades(
    trades: pd.DataFrame,
    pnl_col: str = "pnl",
    time_col: str = "entry_time",
    tz: Optional[str] = None,
) -> pd.Series:
    """
    Collapse a trade log to one P/L observation per calendar date.

    The trade is attributed to the date of `time_col` (entry by default, so the
    observation is aligned with the market move the strategy was exposed to).

    Returns
    -------
    Series indexed by `datetime.date`, named "pnl".
    """
    if pnl_col not in trades.columns:
        raise KeyError(f"pnl column {pnl_col!r} not in trade log")
    if time_col not in trades.columns:
        raise KeyError(f"time column {time_col!r} not in trade log")

    ts = pd.to_datetime(trades[time_col], utc=True, errors="coerce")
    if tz is not None:
        ts = ts.dt.tz_convert(tz)
    valid = ts.notna()

    df = pd.DataFrame({
        "date": ts[valid].dt.date,
        "pnl": pd.to_numeric(trades.loc[valid, pnl_col], errors="coerce"),
    }).dropna()

    out = df.groupby("date")["pnl"].sum().sort_index()
    out.name = "pnl"
    return out


def daily_market_move(
    bars: pd.DataFrame,
    close_col: str = "close",
    tz: Optional[str] = None,
    in_points: bool = True,
) -> pd.Series:
    """
    Per-session move of the traded instrument, from the same bar tape the
    strategy was backtested on.

    The move is measured within the session (last close minus first open-of-day
    close reference), i.e. close-to-close *inside* the day rather than across the
    overnight gap. Intraday strategies are flat overnight, so including the gap
    would attribute an exposure the strategy never had.

    Parameters
    ----------
    bars : DataFrame indexed by timestamp with a close column.
    in_points : if True return the move in index points, else as a fraction.

    Returns
    -------
    Series indexed by `datetime.date`, named "mkt".
    """
    if close_col not in bars.columns:
        raise KeyError(f"close column {close_col!r} not in bars")

    idx = pd.to_datetime(bars.index, utc=True, errors="coerce")
    if tz is not None:
        idx = idx.tz_convert(tz)

    df = pd.DataFrame({
        "date": idx.date,
        "close": pd.to_numeric(bars[close_col], errors="coerce").values,
    }).dropna()

    grp = df.groupby("date")["close"]
    first, last = grp.first(), grp.last()
    move = last - first
    if not in_points:
        move = move / first

    move = move.sort_index()
    move.name = "mkt"
    return move


# ----------------------------------------------------------------------------
# The screen
# ----------------------------------------------------------------------------

@dataclass
class AlphaBetaResult:
    """Outcome of the alpha/beta attribution screen."""
    n_days: int
    n_trades: int
    total_pnl: float
    mean_daily_pnl: float
    frac_long: Optional[float]
    directional: Dict[str, Any]
    with_convexity: Dict[str, Any]
    beta_contracts_equiv: Optional[float]
    pnl_explained_by_beta: float
    pnl_residual_alpha: float
    verdict: str
    notes: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def alpha_beta_screen(
    daily_pnl: pd.Series,
    daily_market: pd.Series,
    include_flat_days: bool = True,
    point_value: Optional[float] = None,
    frac_long: Optional[float] = None,
    n_trades: Optional[int] = None,
    alpha_threshold: float = 0.05,
    hac_lags: Optional[int] = None,
) -> AlphaBetaResult:
    """
    Test whether strategy P/L survives control for the instrument's own move.

    Parameters
    ----------
    daily_pnl : per-date strategy P/L in dollars (from `daily_pnl_from_trades`).
    daily_market : per-date instrument move (from `daily_market_move`).
    include_flat_days : if True, days in the strategy's active range on which it
        did not trade enter the regression with P/L = 0. This is the honest
        equity-curve regression: a strategy that sits out is still making a
        (null) bet. If False, only traded days are used.
    point_value : dollars per index point (MES 5.0, MNQ 2.0). When supplied and
        `daily_market` is in points, beta is also reported as an equivalent net
        contract count, which is the interpretable form.
    frac_long : fraction of trades taken long, carried through for context.
    n_trades : trade count, carried through for context.
    alpha_threshold : p-value below which alpha is called non-zero.
    hac_lags : Newey-West truncation lag; None uses the automatic rule.

    Returns
    -------
    AlphaBetaResult
    """
    pnl = daily_pnl.copy()
    mkt = daily_market.copy()
    notes: list = []

    if len(pnl) == 0:
        raise ValueError("daily_pnl is empty")

    if include_flat_days:
        lo, hi = min(pnl.index), max(pnl.index)
        in_range = mkt.loc[(mkt.index >= lo) & (mkt.index <= hi)]
        pnl = pnl.reindex(in_range.index, fill_value=0.0)
        mkt = in_range
        notes.append(
            f"flat days zero-filled across the strategy's active range "
            f"({lo} to {hi}); {int((pnl == 0).sum())} of {len(pnl)} days flat"
        )
    else:
        common = pnl.index.intersection(mkt.index)
        pnl, mkt = pnl.loc[common], mkt.loc[common]
        notes.append("restricted to traded days only")

    common = pnl.index.intersection(mkt.index)
    pnl, mkt = pnl.loc[common], mkt.loc[common]
    if len(pnl) < 10:
        raise ValueError(
            f"only {len(pnl)} aligned days; need >= 10 for a meaningful regression"
        )

    y = pnl.to_numpy(dtype=float)
    m = mkt.to_numpy(dtype=float)
    ones = np.ones_like(m)

    fit1 = ols_hac(y, np.column_stack([ones, m]), ["alpha", "beta"], hac_lags)
    fit2 = ols_hac(
        y,
        np.column_stack([ones, m, np.abs(m)]),
        ["alpha", "beta", "gamma_abs"],
        hac_lags,
    )

    beta = fit1.get("beta")["coef"]
    alpha = fit1.get("alpha")["coef"]
    beta_contracts = beta / point_value if point_value else None

    # Dollar attribution over the sample.
    explained = float(beta * m.sum())
    residual = float(alpha * len(y))

    a1_p = fit1.get("alpha")["p"]
    a2_p = fit2.get("alpha")["p"]
    a1_sig = a1_p < alpha_threshold and fit1.get("alpha")["coef"] > 0
    a2_sig = a2_p < alpha_threshold and fit2.get("alpha")["coef"] > 0

    if a1_sig and a2_sig:
        verdict = "ALPHA_SURVIVES"
    elif a1_sig and not a2_sig:
        verdict = "CONVEXITY_NOT_ALPHA"
        notes.append(
            "alpha vanishes once |market move| is controlled: the strategy is "
            "paid for range, not for direction-independent skill"
        )
    elif fit1.get("beta")["p"] < alpha_threshold:
        verdict = "BETA_ONLY"
        notes.append(
            "no significant intercept but significant market loading: this is "
            "directional exposure, not edge"
        )
    else:
        verdict = "NO_SIGNAL"
        notes.append("neither alpha nor beta is distinguishable from zero")

    return AlphaBetaResult(
        n_days=int(len(y)),
        n_trades=int(n_trades) if n_trades is not None else -1,
        total_pnl=float(y.sum()),
        mean_daily_pnl=float(y.mean()),
        frac_long=float(frac_long) if frac_long is not None else None,
        directional=fit1.to_dict(),
        with_convexity=fit2.to_dict(),
        beta_contracts_equiv=float(beta_contracts) if beta_contracts is not None else None,
        pnl_explained_by_beta=explained,
        pnl_residual_alpha=residual,
        verdict=verdict,
        notes=notes,
    )


def format_alpha_beta_report(res: AlphaBetaResult, title: str = "") -> str:
    """Human-readable summary of an alpha/beta screen."""
    lines = []
    if title:
        lines += [title, "=" * len(title)]
    lines.append(
        f"days={res.n_days}  trades={res.n_trades}  total_pnl=${res.total_pnl:,.0f}  "
        f"mean_daily=${res.mean_daily_pnl:,.2f}"
    )
    if res.frac_long is not None:
        lines.append(f"long share: {res.frac_long:.1%}")

    for label, spec in (("Spec 1 (directional)", res.directional),
                        ("Spec 2 (+convexity)", res.with_convexity)):
        lines.append(f"\n{label}  R^2={spec['r_squared']:.3f}  HAC lags={spec['hac_lags']}")
        for name, t in spec["terms"].items():
            star = "*" if t["p"] < 0.05 else " "
            lines.append(
                f"  {name:<10} {t['coef']:>12.4f}  se={t['se']:>10.4f}  "
                f"t={t['t']:>7.2f}  p={t['p']:.4f} {star}"
            )

    if res.beta_contracts_equiv is not None:
        lines.append(
            f"\nnet exposure implied by beta: "
            f"{res.beta_contracts_equiv:+.3f} contracts-equivalent"
        )
    lines.append(
        f"P/L attributed to market move: ${res.pnl_explained_by_beta:,.0f}   "
        f"P/L attributed to alpha: ${res.pnl_residual_alpha:,.0f}"
    )
    lines.append(f"\nVERDICT: {res.verdict}")
    for n in res.notes:
        lines.append(f"  - {n}")
    return "\n".join(lines)
