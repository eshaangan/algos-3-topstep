"""Pre-registered validation harness.

The point of this module is *discipline*, not cleverness. A year of this repo's
history shows the failure mode is not a shortage of ideas — it is promoting a
strategy on one lucky month and a Sharpe estimated from ~30 trades. The deployed
ML scalper is the textbook case: April 2026 looked brilliant (WR 58.6%, Sharpe
8.6), May 2026 was dead (WR 40.9%, negative), and both are just noise around a
marginal edge. This module exists to make that mistake mechanically hard.

What it enforces
----------------
1. A **pre-registered decision rule** (``preregister.yaml``) written down BEFORE
   the holdout is touched, and content-hashed so any post-hoc edit to the rule is
   detectable in the audit trail.
2. A **frozen holdout** window evaluated and logged to an append-only ledger
   (``holdout_ledger.jsonl``). Re-touching the holdout, or touching it with a
   changed rule, is flagged loudly — that is the p-hacking detector.
3. A **GO/NO-GO gate** that requires the edge to hold across MULTIPLE months and
   to clear the Deflated Sharpe Ratio (which penalises the number of configs you
   searched), not just post a positive total.

What it deliberately does NOT do
--------------------------------
Strategy logic (model, thresholds, session filter) lives in the *adapter* that
produces the ``signals`` table — never in here. The simulator below is
intentionally simple and explicit so every number is auditable.

The Deflated Sharpe Ratio implemented here is the Bailey & López de Prado (2014)
formula — the same methodology as
``ml_intraday_v3/experiments/diagnostics.py::compute_dsr``. It is vendored (≈40
lines of numpy + stdlib) rather than imported because that module pulls matplotlib
and a cost-curve chain at import time, which has no place in a live-trading gate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yaml

_VALIDATION_DIR = Path(__file__).resolve().parent
DEFAULT_PREREGISTER = _VALIDATION_DIR / "preregister.yaml"
DEFAULT_LEDGER = _VALIDATION_DIR / "holdout_ledger.jsonl"

_NORM = NormalDist()


# ─────────────────────────────────────────────────── deflated sharpe ─────────

def deflated_sharpe_ratio(
    returns: Iterable[float],
    n_trials: int,
    target_sharpe: float = 0.0,
) -> dict:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Accounts for selection bias from testing ``n_trials`` configurations, plus
    the skew/kurtosis and estimation uncertainty of the Sharpe estimate.

    Returns a dict with ``dsr`` in [0, 1] (probability the true Sharpe exceeds
    the selection-bias-inflated benchmark), or ``dsr=None`` with a ``reason`` for
    degenerate inputs. Interpretation: dsr > 0.95 ≈ p < 0.05; dsr < 0.5 ≈ likely
    overfit/luck. Sharpe here is per-trade (non-annualised) — scale-invariant, so
    raw dollar PnL per trade is a valid input.
    """
    r = np.asarray(list(returns), dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return {"dsr": None, "reason": "insufficient_returns", "n_obs": int(r.size)}

    mean = float(np.mean(r))
    std = float(np.std(r, ddof=1))
    if std == 0.0:
        return {"dsr": None, "reason": "zero_variance", "n_obs": int(r.size)}

    sharpe = mean / std
    z = (r - mean) / std
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))
    n = r.size

    sr_var = (1.0 - skew * sharpe + ((kurt - 1.0) / 4.0) * sharpe ** 2) / (n - 1)
    if sr_var <= 0:
        return {"dsr": None, "reason": "nonpositive_sr_variance", "n_obs": int(n)}
    sr_std = float(np.sqrt(sr_var))

    # Expected maximum Sharpe across n_trials independent draws under the null.
    n_trials = max(int(n_trials), 1)
    if n_trials == 1:
        sr_star = target_sharpe
    else:
        emc = 0.5772156649015329  # Euler–Mascheroni
        z1 = _NORM.inv_cdf(1.0 - 1.0 / n_trials)
        z2 = _NORM.inv_cdf(1.0 - 1.0 / (n_trials * np.e))
        sr_star = target_sharpe + sr_std * ((1.0 - emc) * z1 + emc * z2)

    z_score = (sharpe - sr_star) / sr_std
    dsr = float(_NORM.cdf(z_score))
    return {
        "dsr": dsr,
        "sharpe_per_trade": sharpe,
        "sr_star": float(sr_star),
        "sr_std": sr_std,
        "skewness": skew,
        "kurtosis": kurt,
        "n_obs": int(n),
        "n_trials": n_trials,
        "z_score": float(z_score),
    }


# ─────────────────────────────────────────────────── trade simulator ─────────

@dataclass
class SimParams:
    """Execution parameters for the auditable trade simulator.

    Cost models (``cost_model``):
      - ``"latency"`` (DEFAULT): calibrated from live MNQ fills on 2026-07-08.
        Slippage decomposes into (a) half-spread paid by any marketable order and
        (b) adverse drift while the decision pipeline runs, which measured
        ~0.35 pt/s against a ~20 pt 1-min ATR => ``drift_frac_atr_per_sec`` of
        the signal bar's ATR per second of ``decision_lag_secs``. Entries and
        time exits pay (a)+(b); stop exits (exchange-side stop-market) pay (a);
        profit targets (resting limits) pay nothing. Calibration is against
        1-MIN bar ATR — recalibrate the fraction if simulating other bar sizes.
      - ``"fixed"``: legacy flat ``slippage_ticks`` per side on every fill.
        Kept for comparison; it flattered imb_fade_midday by ~$1,100 on a
        9-day holdout, so never use it for a deployment decision.
    """
    pt_atr: float = 2.0
    sl_atr: float = 1.0
    horizon_bars: int = 12
    cooldown_bars: int = 3
    max_trades_per_day: int = 6
    point_value: float = 2.0
    tick_size: float = 0.25
    commission_per_side: float = 0.62
    slippage_ticks: int = 1                    # "fixed" model only
    n_contracts: int = 1
    cost_model: str = "latency"                # "latency" | "fixed"
    decision_lag_secs: float = 3.0             # stream pipeline: ~2s poll + submit
    drift_frac_atr_per_sec: float = 0.0175     # measured 2026-07-08 (1-min bars)
    half_spread_ticks: float = 1.0             # book-crossing cost per marketable order


def simulate(bars: pd.DataFrame, signals: pd.DataFrame, p: SimParams) -> pd.DataFrame:
    """Simulate trades from desired-entry signals. Strategy-agnostic.

    Parameters
    ----------
    bars : DataFrame indexed by tz-aware timestamp, columns open/high/low/close.
    signals : DataFrame aligned to ``bars.index`` with columns:
        - ``direction`` : int in {1, -1, 0}. The entry the strategy WANTS on that
          bar after its own filters (session, threshold, NaN handling). 0 = none.
        - ``atr`` : float, the ATR used to size PT/SL on that bar.
      Any bar with direction==0, NaN atr, or atr<=0 is treated as "no entry".
    p : SimParams.

    Returns a trades DataFrame: entry_time, exit_time, direction, entry_price,
    exit_price, pnl, exit_reason, bars_held. Mirrors the (fixed) live runner:
    stop-first on a bar that spans both stop and target (conservative), 1-tick
    slippage each way, commission both sides, time stop at horizon, cooldown and
    max-trades/day enforced; day boundary by the bar timestamp's calendar date.
    """
    idx = bars.index
    o = bars["open"].to_numpy(float)  # noqa: F841 (kept for clarity/parity)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    direction = signals["direction"].reindex(idx).fillna(0).to_numpy(int)
    atr = signals["atr"].reindex(idx).to_numpy(float)

    comm = 2.0 * p.commission_per_side * p.n_contracts
    scale = p.point_value * p.n_contracts
    if p.cost_model == "fixed":
        fixed = p.slippage_ticks * p.tick_size
        half_spread = fixed
        lag_drift = lambda a: 0.0                     # noqa: E731
        entry_slip = lambda a: fixed                  # noqa: E731
    else:  # "latency" — see SimParams docstring for calibration provenance
        half_spread = p.half_spread_ticks * p.tick_size
        lag_drift = lambda a: p.decision_lag_secs * p.drift_frac_atr_per_sec * a  # noqa: E731
        entry_slip = lambda a: half_spread + lag_drift(a)                         # noqa: E731

    trades: list[dict] = []
    pos: Optional[dict] = None
    cooldown = 0
    trades_today = 0
    cur_day = None

    for i in range(len(idx)):
        day = idx[i].date()
        if day != cur_day:
            cur_day = day
            trades_today = 0

        if pos is not None:
            pos["bars_held"] += 1
            hi, lo, cl = h[i], l[i], c[i]
            d = pos["direction"]
            sl_p, pt_p = pos["stop"], pos["target"]
            stop_hit = (d == 1 and lo <= sl_p) or (d == -1 and hi >= sl_p)
            targ_hit = (d == 1 and hi >= pt_p) or (d == -1 and lo <= pt_p)

            exit_px = None
            reason = None
            if stop_hit:                              # both-hit -> stop (conservative)
                exit_px, reason = sl_p - half_spread * d, "stop_loss"
            elif targ_hit:                            # resting limit: no crossing cost
                pt_slip = half_spread if p.cost_model == "fixed" else 0.0
                exit_px, reason = pt_p - pt_slip * d, "profit_target"
            elif pos["bars_held"] >= p.horizon_bars:  # market out after pipeline lag
                exit_px, reason = cl - (half_spread + lag_drift(pos["atr"])) * d, "time_stop"

            if reason is not None:
                pnl = (exit_px - pos["entry"]) * d * scale - comm
                trades.append({
                    "entry_time": pos["entry_time"],
                    "exit_time": idx[i],
                    "direction": d,
                    "entry_price": pos["entry"],
                    "exit_price": exit_px,
                    "pnl": pnl,
                    "exit_reason": reason,
                    "bars_held": pos["bars_held"],
                })
                pos = None
                cooldown = p.cooldown_bars
            continue

        if cooldown > 0:
            cooldown -= 1
            continue
        if trades_today >= p.max_trades_per_day:
            continue

        d = int(direction[i])
        a = atr[i]
        if d == 0 or not np.isfinite(a) or a <= 0:
            continue

        entry = c[i] + entry_slip(a) * d
        pos = {
            "direction": d,
            "entry": entry,
            "stop": entry - p.sl_atr * a * d,
            "target": entry + p.pt_atr * a * d,
            "bars_held": 0,
            "entry_time": idx[i],
            "atr": a,
        }
        trades_today += 1

    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────── stats ───────────────────

def aggregate_stats(trades: pd.DataFrame) -> dict:
    """Aggregate performance over a trade set (raw dollars; 1× sizing per SimParams)."""
    if trades is None or len(trades) == 0:
        return {"n_trades": 0, "total_pnl": 0.0, "win_rate": 0.0, "max_dd": 0.0,
                "sharpe_per_trade": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "profit_factor": 0.0, "n_days": 0, "exits": {}}
    pnl = trades["pnl"].to_numpy(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    cum = np.cumsum(pnl)
    max_dd = float((cum - np.maximum.accumulate(cum)).min())
    std = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0
    sharpe = float(np.mean(pnl) / std) if std > 0 else 0.0
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    days = pd.to_datetime(trades["entry_time"]).dt.date.nunique()
    return {
        "n_trades": int(len(trades)),
        "n_days": int(days),
        "total_pnl": float(pnl.sum()),
        "win_rate": float((pnl > 0).mean()),
        "max_dd": max_dd,
        "sharpe_per_trade": sharpe,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "exits": trades["exit_reason"].value_counts().to_dict(),
    }


def monthly_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """Per-calendar-month stats. Columns: month, n_trades, win_rate, pnl, max_dd."""
    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=["month", "n_trades", "win_rate", "pnl", "max_dd"])
    t = trades.copy()
    t["month"] = pd.to_datetime(t["entry_time"]).dt.strftime("%Y-%m")
    rows = []
    for m, grp in t.groupby("month"):
        pnl = grp["pnl"].to_numpy(float)
        cum = np.cumsum(pnl)
        rows.append({
            "month": m,
            "n_trades": int(len(grp)),
            "win_rate": float((pnl > 0).mean()),
            "pnl": float(pnl.sum()),
            "max_dd": float((cum - np.maximum.accumulate(cum)).min()),
        })
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


# ─────────────────────────────────────────── pre-registration & gate ─────────

# Fields that constitute the "rule". Changing any of these changes the content
# hash, so a post-hoc edit is visible in the ledger. Descriptive metadata
# (strategy_id, comments) is intentionally excluded from the hash.
_HASHED_FIELDS = ("holdout", "dsr_n_trials", "gate", "sim")


@dataclass
class PreRegistration:
    strategy_id: str
    holdout: dict           # {start: 'YYYY-MM-DD', end: 'YYYY-MM-DD'}
    dsr_n_trials: int
    gate: dict              # decision-rule thresholds
    sim: dict               # SimParams + adapter params (conf, min_atr_pts, ...)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PREREGISTER) -> "PreRegistration":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            strategy_id=raw["strategy_id"],
            holdout=raw["holdout"],
            dsr_n_trials=int(raw["dsr_n_trials"]),
            gate=raw["gate"],
            sim=raw["sim"],
        )

    def content_hash(self) -> str:
        """SHA256 of the canonicalised decision rule. Stable across key ordering."""
        payload = {k: getattr(self, k) for k in _HASHED_FIELDS}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def sim_params(self) -> SimParams:
        keep = {f for f in SimParams().__dataclass_fields__}
        return SimParams(**{k: v for k, v in self.sim.items() if k in keep})


def evaluate(trades: pd.DataFrame, prereg: PreRegistration) -> dict:
    """Apply the pre-registered gate to a holdout trade set. Returns a verdict dict.

    The gate is conjunctive: every condition must pass for a GO. Each failure is
    reported with the observed value vs the threshold, so a NO-GO is explainable.
    """
    agg = aggregate_stats(trades)
    monthly = monthly_breakdown(trades)
    dsr_res = deflated_sharpe_ratio(
        trades["pnl"].to_numpy(float) if len(trades) else [],
        n_trials=prereg.dsr_n_trials,
    )
    g = prereg.gate
    failures: list[str] = []

    if agg["n_trades"] < g["min_oos_trades"]:
        failures.append(f"n_trades {agg['n_trades']} < min_oos_trades {g['min_oos_trades']}")

    n_months = len(monthly)
    if n_months < g["min_months"]:
        failures.append(f"months {n_months} < min_months {g['min_months']}")

    pos_frac = float((monthly["pnl"] > 0).mean()) if n_months else 0.0
    if pos_frac < g["min_positive_month_fraction"]:
        failures.append(
            f"positive-month fraction {pos_frac:.2f} < "
            f"min_positive_month_fraction {g['min_positive_month_fraction']}"
        )

    if g.get("require_net_profitable", True) and agg["total_pnl"] <= 0:
        failures.append(f"total_pnl ${agg['total_pnl']:,.0f} not > 0")

    dsr = dsr_res.get("dsr")
    if dsr is None:
        failures.append(f"DSR unavailable ({dsr_res.get('reason')})")
    elif dsr < g["min_dsr"]:
        failures.append(f"DSR {dsr:.3f} < min_dsr {g['min_dsr']}")

    if "max_drawdown_usd" in g and abs(agg["max_dd"]) > g["max_drawdown_usd"]:
        failures.append(
            f"max_drawdown ${abs(agg['max_dd']):,.0f} > max_drawdown_usd ${g['max_drawdown_usd']:,.0f}"
        )

    return {
        "verdict": "GO" if not failures else "NO-GO",
        "failures": failures,
        "aggregate": agg,
        "monthly": monthly.to_dict(orient="records"),
        "dsr": dsr_res,
        "config_hash": prereg.content_hash(),
        "strategy_id": prereg.strategy_id,
        "holdout": prereg.holdout,
    }


# ─────────────────────────────────────────────────── audit ledger ────────────

class HoldoutLedger:
    """Append-only JSONL record of every holdout evaluation.

    The whole point of a frozen holdout is that it is touched once. This ledger
    makes "touched more than once" and "touched after the rule changed" visible
    rather than silent — the two ways a holdout quietly becomes in-sample.
    """

    def __init__(self, path: Path | str = DEFAULT_LEDGER):
        self.path = Path(path)

    def prior_uses(self, strategy_id: str, holdout: dict) -> list[dict]:
        """Return prior ledger entries for the same strategy + holdout window."""
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (e.get("strategy_id") == strategy_id
                    and e.get("holdout", {}).get("start") == holdout.get("start")
                    and e.get("holdout", {}).get("end") == holdout.get("end")):
                out.append(e)
        return out

    def warnings_for(self, prereg: PreRegistration) -> list[str]:
        """Discipline warnings BEFORE recording a fresh evaluation."""
        prior = self.prior_uses(prereg.strategy_id, prereg.holdout)
        if not prior:
            return []
        warns = [f"Holdout {prereg.holdout} for '{prereg.strategy_id}' has already "
                 f"been evaluated {len(prior)} time(s) — it is no longer a clean holdout."]
        this_hash = prereg.content_hash()
        changed = {e.get("config_hash") for e in prior if e.get("config_hash") != this_hash}
        if changed:
            warns.append("Decision rule has CHANGED since a prior evaluation "
                         "(config hash differs) — this is the classic p-hacking pattern.")
        return warns

    def record(self, prereg: PreRegistration, result: dict) -> dict:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy_id": prereg.strategy_id,
            "holdout": prereg.holdout,
            "config_hash": prereg.content_hash(),
            "verdict": result["verdict"],
            "dsr": result["dsr"].get("dsr"),
            "n_trades": result["aggregate"]["n_trades"],
            "total_pnl": result["aggregate"]["total_pnl"],
            "failures": result["failures"],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry
