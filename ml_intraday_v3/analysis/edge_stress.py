"""
Edge Robustness Stress ("how much of this t-stat is a handful of weeks?")

The third screen, alongside `alpha_beta.py` and `stability.py`. Those ask what
the P/L is made of and whether its distribution drifted. This one asks whether
the headline significance survives dropping any single period, and how
concentrated the P/L is in a few observations.

It exists because of a concrete failure in this project: `weekend_hold_v1` was
carried in the ledger as "t=+4.47, DSR 0.997, 6/7 years positive" and treated as
load-bearing. All of that is arithmetically true. But excluding 2020 (COVID) and
the partial 2026 the t-stat is 2.28, and across 2021-2024 -- 61% of the sample --
the strategy averages $25/week at t=0.84. "6/7 years positive" counted years with
t=0.46 and t=0.54 as wins. No screen in the repo caught that, because none of
them looked.

Three diagnostics:

1. **Leave-one-period-out.** Recompute mean and t with each calendar period
   dropped. An edge that depends on one period is a period, not an edge. The
   reported `worst_loo_t` is the number to judge, not the full-sample t.
2. **Concentration.** Share of total P/L from the top-k observations, and the
   Gini coefficient. High concentration is separately a prop-firm consistency
   problem, not just a statistical one.
3. **Sub-period split.** First half vs second half, with a Welch t-test on the
   difference in means.

None of this is a substitute for out-of-sample testing. It is a cheap way to
find out that an in-sample t-stat was never as strong as it looked.

References
----------
Quant Guild, "Profitable vs Tradable" (2025) -- stability over Sharpe.
Quant Guild, "I Bet You've Never Found Alpha" (2026) -- extend the sample
    through a structural break before believing a backtest.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from scipy import stats


def _t_stat(x: np.ndarray) -> float:
    """One-sample t against zero. NaN when undefined."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return float("nan")
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    return float(x.mean() / sd * np.sqrt(len(x)))


def gini(x: np.ndarray) -> float:
    """
    Gini coefficient of the positive P/L mass, in [0, 1].

    0 means every winning observation contributes equally; values near 1 mean a
    handful of observations carry the strategy. Losses are excluded because the
    question is how concentrated the *earnings* are.
    """
    w = np.sort(np.asarray(x, dtype=float))
    w = w[w > 0]
    n = len(w)
    if n == 0:
        return float("nan")
    if n == 1:
        return 1.0
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * w).sum()) / (n * w.sum()) - (n + 1.0) / n)


@dataclass
class StressResult:
    """Outcome of the robustness stress."""
    n: int
    total_pnl: float
    mean_pnl: float
    full_t: float
    worst_loo_period: Optional[str]
    worst_loo_t: float
    worst_loo_mean: float
    loo_table: List[Dict[str, Any]]
    top_k_share: Dict[str, float]
    gini: float
    first_half_mean: float
    second_half_mean: float
    split_t: float
    split_p: float
    verdict: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def stress_screen(
    pnl: pd.Series,
    period: Optional[pd.Series] = None,
    freq: str = "Y",
    min_period_n: int = 5,
    t_floor: float = 2.0,
    top_k: tuple = (1, 5, 10, 20),
) -> StressResult:
    """
    Stress a P/L series for period dependence and concentration.

    Parameters
    ----------
    pnl : per-trade P/L. If `period` is None the index must be datetime-like,
        and periods are derived from it via `freq`.
    period : optional explicit period label per trade (overrides `freq`).
    freq : pandas period alias used when `period` is None ("Y", "Q", "M").
    min_period_n : periods with fewer trades than this are still dropped in the
        leave-one-out pass but are flagged as thin in the table.
    t_floor : the leave-one-out t-stat a strategy must keep to be called robust.
    top_k : observation counts for the concentration table.

    Returns
    -------
    StressResult
    """
    s = pd.to_numeric(pnl, errors="coerce").dropna()
    if len(s) < 12:
        raise ValueError(f"need >= 12 observations to stress; got {len(s)}")

    if period is not None:
        per = pd.Series(period, index=s.index).reindex(s.index).astype(str)
    else:
        # Guard explicitly. Both pd.to_datetime and pd.DatetimeIndex happily
        # read an integer index as epoch nanoseconds, so a numeric index would
        # silently become 1970 periods instead of raising. Reject numerics
        # outright and only then attempt a parse.
        if isinstance(s.index, pd.DatetimeIndex):
            idx = s.index
        elif pd.api.types.is_numeric_dtype(s.index):
            raise ValueError(
                "pnl index is numeric, not datetime-like; pass `period` explicitly"
            )
        else:
            try:
                idx = pd.DatetimeIndex(s.index)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "pnl index is not datetime-like; pass `period` explicitly"
                ) from exc
        # to_period drops tz; strip it deliberately so the conversion is silent.
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        per = pd.Series(idx.to_period(freq).astype(str), index=s.index)

    x = s.to_numpy(dtype=float)
    full_t = _t_stat(x)
    notes: List[str] = []

    # 1) Leave-one-period-out
    loo: List[Dict[str, Any]] = []
    for p in sorted(per.unique()):
        keep = s[per != p].to_numpy(dtype=float)
        drop = s[per == p]
        if len(keep) < 3:
            continue
        loo.append({
            "period": p,
            "n_dropped": int(len(drop)),
            "dropped_mean": float(drop.mean()),
            "dropped_total": float(drop.sum()),
            "dropped_t": _t_stat(drop.to_numpy(dtype=float)),
            "remaining_n": int(len(keep)),
            "remaining_mean": float(keep.mean()),
            "remaining_t": _t_stat(keep),
            "thin": bool(len(drop) < min_period_n),
        })

    if loo:
        worst = min(loo, key=lambda r: (r["remaining_t"]
                                        if np.isfinite(r["remaining_t"]) else np.inf))
        worst_period, worst_t = worst["period"], worst["remaining_t"]
        worst_mean = worst["remaining_mean"]
    else:
        worst_period, worst_t, worst_mean = None, float("nan"), float("nan")

    # 2) Concentration
    desc = np.sort(x)[::-1]
    total = x.sum()
    shares = {}
    for k in top_k:
        if k <= len(desc):
            shares[f"top_{k}"] = (float(desc[:k].sum() / total)
                                  if total != 0 else float("nan"))
    g = gini(x)

    # 3) First half vs second half
    half = len(x) // 2
    a, b = x[:half], x[half:]
    split_t, split_p = stats.ttest_ind(b, a, equal_var=False)

    # Verdict
    if not np.isfinite(worst_t):
        verdict = "UNKNOWN"
    elif worst_t < t_floor:
        verdict = "PERIOD_DEPENDENT"
        notes.append(
            f"dropping {worst_period} alone takes t from {full_t:.2f} to "
            f"{worst_t:.2f} (below the {t_floor:.1f} floor): the edge leans on "
            f"one period"
        )
    else:
        verdict = "ROBUST"

    if shares.get("top_10", 0) > 0.5:
        notes.append(
            f"top 10 observations carry {shares['top_10']:.0%} of total P/L — "
            f"also a prop-firm consistency problem, not only a statistical one"
        )
    if np.isfinite(g) and g > 0.6:
        notes.append(f"Gini {g:.2f}: earnings are concentrated in few winners")
    if split_p < 0.05:
        direction = "stronger" if b.mean() > a.mean() else "weaker"
        notes.append(
            f"second half is significantly {direction} than the first "
            f"({a.mean():.2f} -> {b.mean():.2f}, p={split_p:.3f})"
        )

    thin = [r["period"] for r in loo if r["thin"]]
    if thin:
        notes.append(f"thin periods (<{min_period_n} trades): {', '.join(thin)}")

    return StressResult(
        n=int(len(x)),
        total_pnl=float(total),
        mean_pnl=float(x.mean()),
        full_t=full_t,
        worst_loo_period=worst_period,
        worst_loo_t=float(worst_t),
        worst_loo_mean=float(worst_mean),
        loo_table=loo,
        top_k_share=shares,
        gini=g,
        first_half_mean=float(a.mean()),
        second_half_mean=float(b.mean()),
        split_t=float(split_t),
        split_p=float(split_p),
        verdict=verdict,
        notes=notes,
    )


def format_stress_report(res: StressResult, title: str = "") -> str:
    """Human-readable summary of a robustness stress."""
    lines = []
    if title:
        lines += [title, "=" * len(title)]
    lines.append(
        f"n={res.n}  total=${res.total_pnl:,.0f}  mean=${res.mean_pnl:,.2f}  "
        f"full-sample t={res.full_t:.2f}"
    )
    lines.append(
        f"WORST leave-one-out: drop {res.worst_loo_period} -> "
        f"t={res.worst_loo_t:.2f}, mean=${res.worst_loo_mean:,.2f}"
    )

    lines.append("\nLeave-one-period-out:")
    lines.append(f"  {'period':<10}{'n':>5}{'drop_mean':>12}{'drop_t':>9}"
                 f"{'rest_mean':>12}{'rest_t':>9}")
    for r in res.loo_table:
        flag = " *thin" if r["thin"] else ""
        lines.append(
            f"  {r['period']:<10}{r['n_dropped']:>5}{r['dropped_mean']:>12.2f}"
            f"{r['dropped_t']:>9.2f}{r['remaining_mean']:>12.2f}"
            f"{r['remaining_t']:>9.2f}{flag}"
        )

    lines.append("\nConcentration:")
    for k, v in res.top_k_share.items():
        lines.append(f"  {k:<8} {v:>7.1%} of total P/L")
    lines.append(f"  gini     {res.gini:>7.2f}")

    lines.append(
        f"\nHalves: {res.first_half_mean:,.2f} -> {res.second_half_mean:,.2f}  "
        f"(Welch t={res.split_t:.2f}, p={res.split_p:.4f})"
    )
    lines.append(f"\nVERDICT: {res.verdict}")
    for n in res.notes:
        lines.append(f"  - {n}")
    return "\n".join(lines)
