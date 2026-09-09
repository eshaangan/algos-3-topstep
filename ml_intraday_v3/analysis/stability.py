"""
P/L Distribution Stability Screen ("profitable vs tradable")

Motivated by Quant Guild lecture 77, "Profitable vs Tradable: Why Most
Strategies Fail Live": the first question about a signal is not its Sharpe, it
is whether its P/L distribution is *stable*. A strategy whose per-trade P/L
distribution drifts across time is not tradable regardless of backtest Sharpe,
because the distribution you sized against is not the one you will realize.

This complements -- it does not replace -- the existing DSR/PBO screens. Those
ask "is this Sharpe real given how many things I tried?". This asks a different
question: "is the P/L generating process the same at the end of the sample as at
the start?". A strategy can pass DSR and fail this, and vice versa.

Method
------
1. Split the trade log chronologically into K equal-count blocks.
2. Bin P/L using quantile edges of the *pooled* sample, so bins are adaptive
   and directly comparable across blocks.
3. Compute Jensen-Shannon divergence between block histograms (Laplace
   smoothed). JS is used as the headline rather than raw KL because it is
   symmetric and bounded in [0, ln 2], so it is comparable across configs with
   different trade counts and P/L scales.
4. Calibrate against a permutation null: shuffle trade order many times and
   recompute. Without this step small-n strategies always look unstable, since
   two 15-trade blocks differ substantially by chance alone. The p-value is the
   share of shuffles whose divergence meets or exceeds the observed one.

Also reported is the Law of Total Expectation decomposition per block,

    E[P/L] = E[P/L | win] * P(win) + E[P/L | loss] * P(loss)

which localises *which component* moved: hit rate, winner size, or loser size.
That is the diagnostic that tells you whether a strategy decayed because its
edge disappeared or because its risk control changed.

References
----------
Quant Guild, "Profitable vs Tradable: Why Most Strategies Fail Live" (2025).
Lin, J. (1991), "Divergence measures based on the Shannon entropy".
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------------
# Divergences on discrete distributions
# ----------------------------------------------------------------------------

def _normalize(counts: np.ndarray, smoothing: float) -> np.ndarray:
    """Laplace-smoothed probability vector."""
    p = np.asarray(counts, dtype=float) + smoothing
    total = p.sum()
    if total <= 0:
        raise ValueError("cannot normalize an all-zero count vector")
    return p / total


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    KL(p || q) in nats. Inputs are probability vectors over the same support.

    Both must be strictly positive; use `_normalize` with smoothing > 0 to
    guarantee that when the inputs come from finite samples.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    if np.any(q <= 0) or np.any(p < 0):
        raise ValueError("q must be strictly positive and p non-negative")
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon divergence in nats: symmetric, bounded in [0, ln 2].

    JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m),  m = 0.5 * (p + q)
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def _histogram(values: np.ndarray, edges: np.ndarray, smoothing: float) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    return _normalize(counts, smoothing)


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Bin edges at pooled quantiles, de-duplicated and widened at the tails so
    every observation falls inside the range.
    """
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(values, qs))
    if len(edges) < 2:
        # Degenerate (all P/L identical): make a trivial two-bin support.
        c = float(values[0]) if len(values) else 0.0
        edges = np.array([c - 0.5, c + 0.5])
    edges = edges.astype(float).copy()
    span = max(float(edges[-1] - edges[0]), 1e-9)
    edges[0] -= span * 1e-6
    edges[-1] += span * 1e-6
    return edges


# ----------------------------------------------------------------------------
# Block statistics
# ----------------------------------------------------------------------------

def _lte_block_stats(pnl: np.ndarray) -> Dict[str, float]:
    """Law of Total Expectation decomposition for one block of trades."""
    n = len(pnl)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    p_win = len(wins) / n if n else 0.0
    return {
        "n": int(n),
        "mean_pnl": float(pnl.mean()) if n else 0.0,
        "total_pnl": float(pnl.sum()) if n else 0.0,
        "p_win": float(p_win),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "std_pnl": float(pnl.std(ddof=1)) if n > 1 else 0.0,
    }


def _mean_pairwise_js(
    blocks: List[np.ndarray], edges: np.ndarray, smoothing: float
) -> tuple:
    """Mean pairwise and first-vs-last JS divergence across blocks."""
    hists = [_histogram(b, edges, smoothing) for b in blocks]
    pairs = []
    for i in range(len(hists)):
        for j in range(i + 1, len(hists)):
            pairs.append(js_divergence(hists[i], hists[j]))
    mean_js = float(np.mean(pairs)) if pairs else 0.0
    first_last = float(js_divergence(hists[0], hists[-1])) if len(hists) > 1 else 0.0
    return mean_js, first_last


def _split_blocks(values: np.ndarray, n_blocks: int) -> List[np.ndarray]:
    """Chronological split into (near) equal-count blocks."""
    return [b for b in np.array_split(values, n_blocks) if len(b) > 0]


# ----------------------------------------------------------------------------
# The screen
# ----------------------------------------------------------------------------

@dataclass
class StabilityResult:
    """Outcome of the P/L distribution stability screen."""
    n_trades: int
    n_blocks: int
    n_bins: int
    mean_pairwise_js: float
    first_last_js: float
    p_value: float
    null_mean_js: float
    null_q95_js: float
    ks_first_last_stat: float
    ks_first_last_p: float
    block_stats: List[Dict[str, float]]
    verdict: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def stability_screen(
    trades: pd.DataFrame,
    pnl_col: str = "pnl",
    time_col: Optional[str] = "entry_time",
    n_blocks: int = 3,
    n_bins: int = 10,
    n_permutations: int = 2000,
    smoothing: float = 0.5,
    alpha: float = 0.05,
    random_state: int = 0,
) -> StabilityResult:
    """
    Test whether the per-trade P/L distribution is stable across time.

    Parameters
    ----------
    trades : trade log.
    pnl_col : per-trade P/L column.
    time_col : timestamp column used to order trades chronologically. If None,
        the existing row order is assumed to already be chronological.
    n_blocks : number of equal-count chronological blocks (>= 2).
    n_bins : target number of pooled-quantile bins.
    n_permutations : shuffles used to build the null. 0 skips the null and
        returns p_value = NaN.
    smoothing : Laplace smoothing added to every bin count.
    alpha : significance level for the drift verdict.
    random_state : seed for the permutation null.

    Returns
    -------
    StabilityResult
    """
    if n_blocks < 2:
        raise ValueError("n_blocks must be at least 2")
    if pnl_col not in trades.columns:
        raise KeyError(f"pnl column {pnl_col!r} not in trade log")

    df = trades.copy()
    if time_col is not None:
        if time_col not in df.columns:
            raise KeyError(f"time column {time_col!r} not in trade log")
        df["_ts"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.dropna(subset=["_ts"]).sort_values("_ts")

    pnl = pd.to_numeric(df[pnl_col], errors="coerce").dropna().to_numpy(dtype=float)
    n = len(pnl)
    notes: List[str] = []

    min_needed = 5 * n_blocks
    if n < min_needed:
        raise ValueError(
            f"only {n} trades; need >= {min_needed} for {n_blocks} blocks "
            f"(at least 5 trades per block)"
        )
    if n < 30:
        notes.append(
            f"n={n} is small; the permutation null is wide and this screen has "
            f"low power to detect drift"
        )

    edges = _quantile_edges(pnl, n_bins)
    n_bins_eff = len(edges) - 1

    blocks = _split_blocks(pnl, n_blocks)
    obs_mean_js, obs_first_last = _mean_pairwise_js(blocks, edges, smoothing)

    # Permutation null: same n, same pooled P/L, chronology destroyed.
    if n_permutations and n_permutations > 0:
        rng = np.random.default_rng(random_state)
        null = np.empty(n_permutations, dtype=float)
        for i in range(n_permutations):
            shuffled = rng.permutation(pnl)
            null[i], _ = _mean_pairwise_js(
                _split_blocks(shuffled, n_blocks), edges, smoothing
            )
        # +1 correction keeps the p-value strictly positive.
        p_value = float((np.sum(null >= obs_mean_js) + 1) / (n_permutations + 1))
        null_mean = float(null.mean())
        null_q95 = float(np.quantile(null, 0.95))
    else:
        p_value, null_mean, null_q95 = float("nan"), float("nan"), float("nan")
        notes.append("permutation null skipped; p-value not available")

    ks_stat, ks_p = stats.ks_2samp(blocks[0], blocks[-1])

    block_stats = [_lte_block_stats(b) for b in blocks]

    if not np.isnan(p_value) and p_value < alpha:
        verdict = "UNSTABLE"
        notes.append(
            "P/L distribution drifts more than chance: backtest distribution is "
            "not the one you would trade forward"
        )
    elif not np.isnan(p_value):
        verdict = "STABLE"
    else:
        verdict = "UNKNOWN"

    # Localise the drift for the reader.
    if len(block_stats) >= 2:
        first, last = block_stats[0], block_stats[-1]
        for label, key in (("hit rate", "p_win"),
                           ("average winner", "avg_win"),
                           ("average loser", "avg_loss")):
            a, b = first[key], last[key]
            if abs(a) > 1e-12 and abs(b - a) / abs(a) > 0.25:
                notes.append(
                    f"{label} moved {a:.4g} -> {b:.4g} from first to last block"
                )

    return StabilityResult(
        n_trades=n,
        n_blocks=len(blocks),
        n_bins=int(n_bins_eff),
        mean_pairwise_js=obs_mean_js,
        first_last_js=obs_first_last,
        p_value=p_value,
        null_mean_js=null_mean,
        null_q95_js=null_q95,
        ks_first_last_stat=float(ks_stat),
        ks_first_last_p=float(ks_p),
        block_stats=block_stats,
        verdict=verdict,
        notes=notes,
    )


def format_stability_report(res: StabilityResult, title: str = "") -> str:
    """Human-readable summary of a stability screen."""
    lines = []
    if title:
        lines += [title, "=" * len(title)]
    lines.append(
        f"trades={res.n_trades}  blocks={res.n_blocks}  bins={res.n_bins}"
    )
    lines.append(
        f"mean pairwise JS = {res.mean_pairwise_js:.4f} nats   "
        f"(null mean {res.null_mean_js:.4f}, null q95 {res.null_q95_js:.4f})"
    )
    lines.append(f"first-vs-last JS = {res.first_last_js:.4f}")
    lines.append(f"permutation p-value = {res.p_value:.4f}")
    lines.append(
        f"KS(first, last): D={res.ks_first_last_stat:.3f}  p={res.ks_first_last_p:.4f}"
    )

    lines.append("\nLaw of Total Expectation by block:")
    lines.append(
        f"  {'blk':<4}{'n':>5}{'mean':>10}{'P(win)':>9}"
        f"{'avg_win':>10}{'avg_loss':>10}{'total':>11}"
    )
    for i, b in enumerate(res.block_stats):
        lines.append(
            f"  {i:<4}{b['n']:>5}{b['mean_pnl']:>10.2f}{b['p_win']:>9.3f}"
            f"{b['avg_win']:>10.2f}{b['avg_loss']:>10.2f}{b['total_pnl']:>11.2f}"
        )

    lines.append(f"\nVERDICT: {res.verdict}")
    for n in res.notes:
        lines.append(f"  - {n}")
    return "\n".join(lines)
