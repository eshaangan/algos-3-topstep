"""SBDR (Sequential Burst Diffusion Reverter) Backtest.

Strategy: Fade extreme multi-horizon burst Z-scores using wick asymmetry
confirmation on MES 1-min RTH data.

Training: Jun–Dec 2025 (mes_1m_bars_cache.h5)
OOS:      Jan–Feb 2026 (jan_feb_2026_oos_test_1m.h5)

Run:
    python rule_based_v1/diagnostics/sbdr_backtest.py
"""

from __future__ import annotations

import json
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
DIAG = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Instrument constants  (MES)
# ---------------------------------------------------------------------------
POINT_VALUE   = 5.0    # MES: $5 per point
TICK_SIZE     = 0.25
TICK_VALUE    = POINT_VALUE * TICK_SIZE   # $1.25
COMMISSION    = 0.62   # per side per contract (round-turn = 1.24)
N_CONTRACTS   = 1
DAILY_LOSS_LIMIT = 450.0   # hard intraday floor


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_rth(path: Path, key: str, has_ts_col: bool) -> pd.DataFrame:
    with pd.HDFStore(str(path), "r") as s:
        raw = s[key]
    if has_ts_col:
        raw = raw.set_index("timestamp")
    raw.index = pd.to_datetime(raw.index, utc=True).tz_convert("US/Eastern")
    raw = raw.sort_index()
    # RTH 09:30–15:59
    rth = raw[
        (raw.index.hour > 9) | ((raw.index.hour == 9) & (raw.index.minute >= 30))
    ]
    rth = rth[rth.index.hour < 16]
    return rth[["open", "high", "low", "close", "volume"]].copy()


def load_training() -> pd.DataFrame:
    return _load_rth(
        ROOT / "data/processed/mes_1m_bars_cache.h5",
        "/bars_1m",
        has_ts_col=True,
    )


def load_oos() -> pd.DataFrame:
    with pd.HDFStore(
        str(ROOT / "data/processed/jan_feb_2026_oos_test_1m.h5"), "r"
    ) as s:
        keys = list(s.keys())
        raw = s[keys[0]]
    raw.index = pd.to_datetime(raw.index, utc=True).tz_convert("US/Eastern")
    raw = raw.sort_index()
    rth = raw[
        (raw.index.hour > 9) | ((raw.index.hour == 9) & (raw.index.minute >= 30))
    ]
    rth = rth[rth.index.hour < 16]
    return rth[["open", "high", "low", "close", "volume"]].copy()


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def compute_sigma_hat(
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    closes: np.ndarray,
    alpha: float,
) -> float:
    """Hybrid volatility estimator over a window of bars."""
    u = highs - np.maximum(opens, closes)
    l = np.minimum(opens, closes) - lows
    sigma2_wick = np.mean((u ** 2 + l ** 2) / 2.0)

    sigma2_range = np.mean((highs - lows) ** 2) / (4.0 * math.log(2))

    sigma2_hat = alpha * sigma2_wick + (1.0 - alpha) * sigma2_range
    return math.sqrt(max(sigma2_hat, 1e-8))


def compute_burst_z(
    closes: np.ndarray,
    sigma_hat: float,
    horizons: list[int],
    lam: float,
) -> tuple[float, float, float, int]:
    """Compute Z_min, Z_max, and h_star from multi-horizon burst scores."""
    z_vals = []
    for h in horizons:
        if len(closes) < h + 1:
            z_vals.append(0.0)
            continue
        weights = np.array([math.exp(-lam * i) for i in range(h)])
        denom_sq = np.sum(weights ** 2)
        returns = np.diff(closes[-(h + 1) :])[::-1]  # most-recent first
        z_h = np.dot(weights, returns) / (sigma_hat * math.sqrt(denom_sq))
        z_vals.append(z_h)

    z_arr = np.array(z_vals)
    z_min = float(np.min(z_arr))
    z_max = float(np.max(z_arr))
    abs_arr = np.abs(z_arr)
    h_star = horizons[int(np.argmax(abs_arr))]
    return z_min, z_max, h_star


def compute_wick_asymmetry(
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    closes: np.ndarray,
) -> float:
    """W_t over a window of m bars."""
    u = highs - np.maximum(opens, closes)
    l = np.minimum(opens, closes) - lows
    u2 = np.sum(u ** 2)
    l2 = np.sum(l ** 2)
    return (u2 - l2) / (u2 + l2 + 1e-10)


# ---------------------------------------------------------------------------
# Single-trade result
# ---------------------------------------------------------------------------

@dataclass
class TradeResult:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int        # +1 long, -1 short
    entry_price: float
    exit_price: float
    exit_reason: str
    pnl: float
    z_trigger: float      # |Z| that triggered entry
    delta_w: float        # |ΔW| at entry
    sigma_hat: float
    h_star: int
    date: pd.Timestamp


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

@dataclass
class SBDRConfig:
    vol_window: int = 20
    horizons: list = field(default_factory=lambda: [3, 5, 8])
    lam: float = 0.3
    alpha: float = 0.4
    wick_window: int = 5
    percentile_window: int = 60
    u_percentile: float = 82.0
    u_floor: float = 1.5
    time_stop_bars: int = 5
    entry_start: tuple = (9, 35)
    entry_end: tuple = (15, 40)
    force_flat: tuple = (15, 55)
    daily_loss_limit: float = DAILY_LOSS_LIMIT
    max_daily_losses: int = 3


def _bar_time_ok(ts: pd.Timestamp, cfg: SBDRConfig) -> bool:
    t = (ts.hour, ts.minute)
    return cfg.entry_start <= t <= cfg.entry_end


def run_backtest(bars: pd.DataFrame, cfg: SBDRConfig) -> list[TradeResult]:
    """Run SBDR backtest on a bar DataFrame (already RTH, sorted)."""
    o = bars["open"].values
    h = bars["high"].values
    l = bars["low"].values
    c = bars["close"].values
    ts = bars.index

    n = len(bars)
    warmup = cfg.vol_window + max(cfg.horizons) + 5
    trades: list[TradeResult] = []

    # Rolling buffer for |Z_max_abs| for dynamic threshold
    z_max_abs_hist: list[float] = []

    # Per-session state
    session_date: Optional[pd.Timestamp] = None
    daily_pnl: float = 0.0
    daily_losses: int = 0
    cooldown_until: int = -1

    # Open position state
    in_trade: bool = False
    direction: int = 0
    entry_price: float = 0.0
    entry_bar: int = 0
    stop_dist: float = 0.0
    target1: float = 0.0
    entry_time: pd.Timestamp = ts[0]
    entry_z_trigger: float = 0.0
    entry_delta_w: float = 0.0
    entry_sigma: float = 0.0
    entry_h_star: int = 3

    prev_w: float = 0.0

    for i in range(warmup, n):
        bar_date = ts[i].normalize()

        # Session reset
        if bar_date != session_date:
            # Force flat if still in trade at EOD (shouldn't happen after 15:55 check)
            if in_trade:
                exit_p = c[i - 1]
                raw_pnl = direction * (exit_p - entry_price) * N_CONTRACTS * POINT_VALUE
                comm = 2.0 * COMMISSION * N_CONTRACTS
                pnl = raw_pnl - comm
                trades.append(
                    TradeResult(
                        entry_time=entry_time,
                        exit_time=ts[i - 1],
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=exit_p,
                        exit_reason="session_end",
                        pnl=pnl,
                        z_trigger=entry_z_trigger,
                        delta_w=entry_delta_w,
                        sigma_hat=entry_sigma,
                        h_star=entry_h_star,
                        date=session_date,
                    )
                )
                in_trade = False
            session_date = bar_date
            daily_pnl = 0.0
            daily_losses = 0
            prev_w = 0.0

        # Force flat at 15:55
        if in_trade:
            bar_t = (ts[i].hour, ts[i].minute)
            if bar_t >= cfg.force_flat:
                exit_p = c[i]
                raw_pnl = direction * (exit_p - entry_price) * N_CONTRACTS * POINT_VALUE
                comm = 2.0 * COMMISSION * N_CONTRACTS
                pnl = raw_pnl - comm
                daily_pnl += pnl
                if pnl < 0:
                    daily_losses += 1
                trades.append(
                    TradeResult(
                        entry_time=entry_time,
                        exit_time=ts[i],
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=exit_p,
                        exit_reason="force_flat",
                        pnl=pnl,
                        z_trigger=entry_z_trigger,
                        delta_w=entry_delta_w,
                        sigma_hat=entry_sigma,
                        h_star=entry_h_star,
                        date=session_date,
                    )
                )
                in_trade = False
                continue

        # Compute indicators
        vw = cfg.vol_window
        ww = cfg.wick_window

        sigma_hat = compute_sigma_hat(
            h[i - vw : i], l[i - vw : i], o[i - vw : i], c[i - vw : i],
            cfg.alpha,
        )

        z_min, z_max, h_star = compute_burst_z(
            c[: i + 1], sigma_hat, cfg.horizons, cfg.lam
        )

        z_max_abs = max(abs(z_min), abs(z_max))
        z_max_abs_hist.append(z_max_abs)
        if len(z_max_abs_hist) > cfg.percentile_window:
            z_max_abs_hist.pop(0)

        u_t = max(
            np.percentile(z_max_abs_hist, cfg.u_percentile),
            cfg.u_floor,
        )

        w_t = compute_wick_asymmetry(
            h[i - ww : i], l[i - ww : i], o[i - ww : i], c[i - ww : i]
        )

        # ---- Exit check for open trade ----
        if in_trade:
            bars_held = i - entry_bar
            exited = False

            if direction == 1:  # long
                if l[i] <= entry_price - stop_dist:
                    exit_p = entry_price - stop_dist
                    reason = "stop_loss"
                    exited = True
                elif h[i] >= entry_price + target1:
                    exit_p = entry_price + target1
                    reason = "take_profit"
                    exited = True
                elif bars_held >= cfg.time_stop_bars:
                    exit_p = c[i]
                    reason = "time_stop"
                    exited = True
            else:  # short
                if h[i] >= entry_price + stop_dist:
                    exit_p = entry_price + stop_dist
                    reason = "stop_loss"
                    exited = True
                elif l[i] <= entry_price - target1:
                    exit_p = entry_price - target1
                    reason = "take_profit"
                    exited = True
                elif bars_held >= cfg.time_stop_bars:
                    exit_p = c[i]
                    reason = "time_stop"
                    exited = True

            if exited:
                raw_pnl = direction * (exit_p - entry_price) * N_CONTRACTS * POINT_VALUE
                comm = 2.0 * COMMISSION * N_CONTRACTS
                pnl = raw_pnl - comm
                daily_pnl += pnl
                if pnl < 0:
                    daily_losses += 1
                    cooldown_until = i + 1  # 1-bar cooldown after loss (simple)
                trades.append(
                    TradeResult(
                        entry_time=entry_time,
                        exit_time=ts[i],
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=exit_p,
                        exit_reason=reason,
                        pnl=pnl,
                        z_trigger=entry_z_trigger,
                        delta_w=entry_delta_w,
                        sigma_hat=entry_sigma,
                        h_star=entry_h_star,
                        date=session_date,
                    )
                )
                in_trade = False

        # ---- Entry check ----
        if (
            not in_trade
            and _bar_time_ok(ts[i], cfg)
            and i > cooldown_until
            and daily_losses < cfg.max_daily_losses
            and daily_pnl > -cfg.daily_loss_limit
        ):
            setup_range = h[i] - l[i]
            stop_dist_raw = 0.6 * setup_range
            stop_dist_entry = max(stop_dist_raw, 4.0 * TICK_SIZE)  # floor 4 ticks

            # Long entry
            if z_min < -u_t:
                close_pos = c[i] - l[i]
                bar_range = h[i] - l[i]
                if bar_range > 0 and close_pos >= 0.25 * bar_range:
                    if w_t > prev_w:
                        direction = 1
                        entry_price = c[i] + TICK_SIZE  # 1 tick slippage
                        entry_bar = i
                        entry_time = ts[i]
                        entry_z_trigger = abs(z_min)
                        entry_delta_w = abs(w_t - prev_w)
                        entry_sigma = sigma_hat
                        entry_h_star = h_star
                        stop_dist = stop_dist_entry
                        target1 = 0.8 * sigma_hat * math.sqrt(h_star)
                        in_trade = True

            # Short entry (only if no long triggered)
            elif not in_trade and z_max > u_t:
                close_pos = c[i] - l[i]
                bar_range = h[i] - l[i]
                if bar_range > 0 and close_pos <= 0.75 * bar_range:
                    if w_t < prev_w:
                        direction = -1
                        entry_price = c[i] - TICK_SIZE  # 1 tick slippage
                        entry_bar = i
                        entry_time = ts[i]
                        entry_z_trigger = abs(z_max)
                        entry_delta_w = abs(w_t - prev_w)
                        entry_sigma = sigma_hat
                        entry_h_star = h_star
                        stop_dist = stop_dist_entry
                        target1 = 0.8 * sigma_hat * math.sqrt(h_star)
                        in_trade = True

        prev_w = w_t

    return trades


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def compute_stats(trades: list[TradeResult], label: str = "") -> dict:
    if not trades:
        return {"label": label, "N": 0}

    pnls = np.array([t.pnl for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    # Daily PnL
    daily: dict[pd.Timestamp, float] = defaultdict(float)
    for t in trades:
        daily[t.date] += t.pnl
    daily_vals = np.array(list(daily.values()))
    n_days = len(daily_vals)

    sharpe = 0.0
    if n_days > 1 and daily_vals.std() > 0:
        sharpe = (daily_vals.mean() / daily_vals.std()) * math.sqrt(252)

    trades_per_day = len(trades) / n_days if n_days > 0 else 0.0

    # Max intraday dd
    worst_session = float(min(daily_vals)) if len(daily_vals) > 0 else 0.0

    return {
        "label": label,
        "N": len(trades),
        "n_days": n_days,
        "trades_per_day": round(trades_per_day, 2),
        "wr": round(len(wins) / len(pnls) * 100, 1),
        "avg_win": round(float(wins.mean()), 2) if len(wins) > 0 else 0.0,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) > 0 else 0.0,
        "total_pnl": round(float(pnls.sum()), 2),
        "sharpe": round(sharpe, 3),
        "max_intraday_dd": round(worst_session, 2),
        "max_trade_loss": round(float(pnls.min()), 2),
    }


# ---------------------------------------------------------------------------
# Grid sweep
# ---------------------------------------------------------------------------

def grid_sweep(bars: pd.DataFrame) -> tuple[dict, list[dict]]:
    u_percentiles = [75, 80, 82, 85, 88]
    alphas = [0.3, 0.4, 0.5]
    lambdas = [0.2, 0.3, 0.4]

    results = []
    total = len(u_percentiles) * len(alphas) * len(lambdas)
    done = 0

    print(f"Running grid sweep: {total} combos...")

    for u_pct, alp, lam in product(u_percentiles, alphas, lambdas):
        cfg = SBDRConfig(u_percentile=u_pct, alpha=alp, lam=lam)
        trades = run_backtest(bars, cfg)
        stats = compute_stats(trades, label=f"u{u_pct}_a{alp}_l{lam}")
        stats["u_percentile"] = u_pct
        stats["alpha"] = alp
        stats["lambda"] = lam
        results.append(stats)
        done += 1
        if done % 10 == 0:
            print(f"  {done}/{total} done...")

    results.sort(key=lambda x: x.get("sharpe", -999), reverse=True)
    best = results[0]
    return best, results


# ---------------------------------------------------------------------------
# Monotonicity validation
# ---------------------------------------------------------------------------

def monotonicity_analysis(trades: list[TradeResult], u_t_col: bool = False) -> dict:
    """Bucket by |Z_trigger| and delta_w."""
    if not trades:
        return {}

    # Z-trigger buckets: relative to threshold
    # We'll use absolute quantiles since we don't have per-trade u_t stored.
    z_vals = np.array([t.z_trigger for t in trades])
    pnls = np.array([t.pnl for t in trades])
    wins = pnls > 0

    # Use percentile-based buckets of z_trigger distribution
    p33 = np.percentile(z_vals, 33)
    p66 = np.percentile(z_vals, 66)

    buckets_z = {
        "B1_low": [],
        "B2_mid": [],
        "B3_high": [],
    }
    for t in trades:
        if t.z_trigger < p33:
            buckets_z["B1_low"].append(t)
        elif t.z_trigger < p66:
            buckets_z["B2_mid"].append(t)
        else:
            buckets_z["B3_high"].append(t)

    z_mono = {}
    for k, bk in buckets_z.items():
        if bk:
            bp = np.array([t.pnl for t in bk])
            z_mono[k] = {
                "N": len(bk),
                "wr": round(float(np.mean(bp > 0)) * 100, 1),
                "avg_pnl": round(float(bp.mean()), 2),
            }
        else:
            z_mono[k] = {"N": 0, "wr": 0, "avg_pnl": 0}

    # Delta-W buckets
    dw_vals = np.array([t.delta_w for t in trades])
    buckets_w = {"A_weak": [], "B_mod": [], "C_strong": []}
    for t in trades:
        if t.delta_w < 0.05:
            buckets_w["A_weak"].append(t)
        elif t.delta_w < 0.15:
            buckets_w["B_mod"].append(t)
        else:
            buckets_w["C_strong"].append(t)

    w_mono = {}
    for k, bk in buckets_w.items():
        if bk:
            bp = np.array([t.pnl for t in bk])
            w_mono[k] = {
                "N": len(bk),
                "wr": round(float(np.mean(bp > 0)) * 100, 1),
                "avg_pnl": round(float(bp.mean()), 2),
            }
        else:
            w_mono[k] = {"N": 0, "wr": 0, "avg_pnl": 0}

    # Check monotonicity
    z_wrs = [z_mono[k]["wr"] for k in ["B1_low", "B2_mid", "B3_high"] if z_mono[k]["N"] > 0]
    z_pnls = [z_mono[k]["avg_pnl"] for k in ["B1_low", "B2_mid", "B3_high"] if z_mono[k]["N"] > 0]
    z_wr_mono = all(z_wrs[i] <= z_wrs[i + 1] for i in range(len(z_wrs) - 1))
    z_pnl_mono = all(z_pnls[i] <= z_pnls[i + 1] for i in range(len(z_pnls) - 1))

    w_wrs = [w_mono[k]["wr"] for k in ["A_weak", "B_mod", "C_strong"] if w_mono[k]["N"] > 0]
    w_wr_mono = all(w_wrs[i] <= w_wrs[i + 1] for i in range(len(w_wrs) - 1))

    return {
        "z_buckets": z_mono,
        "w_buckets": w_mono,
        "z_wr_monotone": z_wr_mono,
        "z_pnl_monotone": z_pnl_mono,
        "w_wr_monotone": w_wr_mono,
        "z_mono_pass": z_wr_mono or z_pnl_mono,
    }


# ---------------------------------------------------------------------------
# Per-day analysis
# ---------------------------------------------------------------------------

def per_day_analysis(trades: list[TradeResult]) -> dict:
    if not trades:
        return {}

    daily_trades: dict[pd.Timestamp, list] = defaultdict(list)
    daily_pnl: dict[pd.Timestamp, float] = defaultdict(float)
    for t in trades:
        daily_trades[t.date].append(t)
        daily_pnl[t.date] += t.pnl

    counts = sorted([len(v) for v in daily_trades.values()])
    pnl_vals = sorted(daily_pnl.items(), key=lambda x: x[1])

    long_trades = [t for t in trades if t.direction == 1]
    short_trades = [t for t in trades if t.direction == -1]

    def side_stats(side_trades):
        if not side_trades:
            return {"N": 0, "wr": 0, "avg_pnl": 0}
        p = np.array([t.pnl for t in side_trades])
        return {
            "N": len(p),
            "wr": round(float(np.mean(p > 0)) * 100, 1),
            "avg_pnl": round(float(p.mean()), 2),
        }

    return {
        "trades_per_day": {
            "min": counts[0],
            "p25": int(np.percentile(counts, 25)),
            "median": int(np.median(counts)),
            "p75": int(np.percentile(counts, 75)),
            "max": counts[-1],
        },
        "worst_5_sessions": [
            {"date": str(d.date()), "pnl": round(p, 2)} for d, p in pnl_vals[:5]
        ],
        "best_5_sessions": [
            {"date": str(d.date()), "pnl": round(p, 2)} for d, p in pnl_vals[-5:][::-1]
        ],
        "long": side_stats(long_trades),
        "short": side_stats(short_trades),
    }


# ---------------------------------------------------------------------------
# Monte Carlo combine estimate
# ---------------------------------------------------------------------------

def monte_carlo_combine(sbdr_daily: dict[pd.Timestamp, float], n_paths: int = 10_000) -> dict:
    """
    Sample from SBDR daily PnLs only (portfolio context referenced in spec
    requires other strategies; use SBDR daily PnL distribution as the
    increment, scaled to represent 1 MES contract contribution).
    """
    daily_vals = np.array(list(sbdr_daily.values()))
    if len(daily_vals) < 5:
        return {"error": "insufficient data"}

    rng = np.random.default_rng(42)
    STARTING_EQUITY = 50_000.0
    PASS_TARGET = 53_000.0
    TRAIL_DD = 2_000.0
    DAILY_LOSS_CAP = 1_000.0
    N_DAYS = 60

    pass_count = 0
    bust_count = 0
    pass_days = []

    for _ in range(n_paths):
        equity = STARTING_EQUITY
        peak = STARTING_EQUITY
        passed = False
        busted = False

        sample = rng.choice(daily_vals, size=N_DAYS, replace=True)

        for day_idx, dpnl in enumerate(sample):
            # Cap daily loss
            dpnl = max(dpnl, -DAILY_LOSS_CAP)
            equity += dpnl
            if equity > peak:
                peak = equity

            if equity <= peak - TRAIL_DD or equity <= STARTING_EQUITY - TRAIL_DD:
                busted = True
                break

            if equity >= PASS_TARGET and not passed:
                passed = True
                pass_days.append(day_idx + 1)
                break

        if busted:
            bust_count += 1
        elif passed:
            pass_count += 1

    return {
        "n_paths": n_paths,
        "p_pass_pct": round(pass_count / n_paths * 100, 1),
        "p_bust_pct": round(bust_count / n_paths * 100, 1),
        "median_days_to_pass": round(float(np.median(pass_days)), 1) if pass_days else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("SBDR (Sequential Burst Diffusion Reverter) — Backtest")
    print("=" * 70)

    # ---- Load data ----
    print("\n[1] Loading data...")
    train_bars = load_training()
    oos_bars = load_oos()
    print(f"  Training: {train_bars.index.min().date()} → {train_bars.index.max().date()}"
          f" ({train_bars.index.normalize().nunique()} days, {len(train_bars):,} bars)")
    print(f"  OOS:      {oos_bars.index.min().date()} → {oos_bars.index.max().date()}"
          f" ({oos_bars.index.normalize().nunique()} days, {len(oos_bars):,} bars)")

    # ---- Grid sweep ----
    print("\n[2] Grid Sweep (training data)")
    print("-" * 70)
    best_cfg_dict, all_results = grid_sweep(train_bars)
    print(f"\nTop 15 configurations by Sharpe:")
    print(f"{'Rank':<5} {'Config':<25} {'N':>5} {'N/day':>6} {'WR%':>6} "
          f"{'AvgW':>8} {'AvgL':>8} {'PnL':>9} {'Sharpe':>7} {'MaxIDDD':>9}")
    print("-" * 90)
    for rank, r in enumerate(all_results[:15], 1):
        if r["N"] == 0:
            continue
        print(
            f"{rank:<5} {r['label']:<25} {r['N']:>5} {r['trades_per_day']:>6.2f} "
            f"{r['wr']:>6.1f} {r['avg_win']:>8.2f} {r['avg_loss']:>8.2f} "
            f"{r['total_pnl']:>9.2f} {r['sharpe']:>7.3f} {r['max_intraday_dd']:>9.2f}"
        )

    # Extract best config
    best = best_cfg_dict
    print(f"\nBest config: u_percentile={best['u_percentile']}, "
          f"alpha={best['alpha']}, lambda={best['lambda']}")

    best_cfg = SBDRConfig(
        u_percentile=best["u_percentile"],
        alpha=best["alpha"],
        lam=best["lambda"],
    )

    # ---- Full training run with best config ----
    print("\n[3] Full Training Run (best config)")
    print("-" * 70)
    train_trades = run_backtest(train_bars, best_cfg)
    train_stats = compute_stats(train_trades, label="training")

    for k, v in train_stats.items():
        if k not in ("label",):
            print(f"  {k:<25}: {v}")

    # ---- Monotonicity analysis ----
    print("\n[4] Monotonicity Validation")
    print("-" * 70)
    mono = monotonicity_analysis(train_trades)

    if mono:
        print("\n  Z-trigger buckets (33rd/66th pct splits):")
        print(f"  {'Bucket':<12} {'N':>5} {'WR%':>7} {'AvgPnL':>9}")
        for k in ["B1_low", "B2_mid", "B3_high"]:
            b = mono["z_buckets"].get(k, {})
            print(f"  {k:<12} {b.get('N', 0):>5} {b.get('wr', 0):>7.1f} {b.get('avg_pnl', 0):>9.2f}")
        print(f"  Z WR monotone: {mono['z_wr_monotone']} | Z PnL monotone: {mono['z_pnl_monotone']}")
        print(f"  Mono PASS: {mono['z_mono_pass']}")

        print("\n  Wick asymmetry (ΔW) buckets:")
        print(f"  {'Bucket':<12} {'N':>5} {'WR%':>7} {'AvgPnL':>9}")
        for k in ["A_weak", "B_mod", "C_strong"]:
            b = mono["w_buckets"].get(k, {})
            print(f"  {k:<12} {b.get('N', 0):>5} {b.get('wr', 0):>7.1f} {b.get('avg_pnl', 0):>9.2f}")
        print(f"  W WR monotone: {mono['w_wr_monotone']}")

    # ---- OOS validation ----
    print("\n[5] OOS Validation (Jan–Feb 2026)")
    print("-" * 70)
    oos_trades = run_backtest(oos_bars, best_cfg)
    oos_stats = compute_stats(oos_trades, label="oos")
    for k, v in oos_stats.items():
        if k not in ("label",):
            print(f"  {k:<25}: {v}")

    # WR degradation
    if train_stats["N"] > 0 and oos_stats["N"] > 0:
        wr_diff = abs(oos_stats["wr"] - train_stats["wr"])
        print(f"\n  Training WR: {train_stats['wr']}% | OOS WR: {oos_stats['wr']}% "
              f"| Δ={wr_diff:.1f}pp (limit: 8pp)")

    # ---- Per-day analysis ----
    print("\n[6] Per-Day Analysis (training)")
    print("-" * 70)
    day_ana = per_day_analysis(train_trades)
    if day_ana:
        tpd = day_ana["trades_per_day"]
        print(f"  Trades/day: min={tpd['min']} p25={tpd['p25']} "
              f"median={tpd['median']} p75={tpd['p75']} max={tpd['max']}")

        print("\n  Best 5 sessions:")
        for s in day_ana["best_5_sessions"]:
            print(f"    {s['date']}  ${s['pnl']:>8.2f}")
        print("\n  Worst 5 sessions:")
        for s in day_ana["worst_5_sessions"]:
            print(f"    {s['date']}  ${s['pnl']:>8.2f}")

        print("\n  Long vs Short split:")
        for side, stats in [("LONG", day_ana["long"]), ("SHORT", day_ana["short"])]:
            print(f"    {side}: N={stats['N']} WR={stats['wr']}% avg_pnl=${stats['avg_pnl']:.2f}")

    # ---- Monte Carlo ----
    print("\n[7] Monte Carlo Combine Estimate (SBDR only, 10k paths × 60 days)")
    print("-" * 70)
    train_daily: dict[pd.Timestamp, float] = defaultdict(float)
    for t in train_trades:
        train_daily[t.date] += t.pnl

    mc = monte_carlo_combine(dict(train_daily))
    print(f"  P(pass):         {mc.get('p_pass_pct', 'N/A')}%")
    print(f"  P(bust):         {mc.get('p_bust_pct', 'N/A')}%")
    print(f"  Median days:     {mc.get('median_days_to_pass', 'N/A')}")

    # ---- VERDICT ----
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    oos_wr_diff = abs(oos_stats.get("wr", 0) - train_stats.get("wr", 0)) if oos_stats["N"] > 0 else 999
    z_mono_pass = mono.get("z_mono_pass", False) if mono else False

    n_per_day = train_stats.get("trades_per_day", 0)
    train_wr = train_stats.get("wr", 0)

    if (
        n_per_day >= 1.5
        and train_wr >= 55.0
        and z_mono_pass
        and oos_wr_diff <= 8.0
    ):
        verdict = "PASS"
    elif n_per_day >= 1.0 and train_wr >= 52.0 and z_mono_pass:
        verdict = "MARGINAL"
    else:
        verdict = "KILL"

    print(f"\n  N/day:          {n_per_day:.2f}  (need >= 1.5 for PASS, >= 1.0 for MARGINAL)")
    print(f"  Training WR:    {train_wr:.1f}%  (need >= 55% for PASS, >= 52% for MARGINAL)")
    print(f"  Z-monotonicity: {z_mono_pass}  (must hold)")
    print(f"  OOS WR diff:    {oos_wr_diff:.1f}pp  (need <= 8pp for PASS)")
    print(f"\n  >>> VERDICT: {verdict} <<<")

    # ---- Save results ----
    results = {
        "best_config": {
            "u_percentile": best["u_percentile"],
            "alpha": best["alpha"],
            "lambda": best["lambda"],
        },
        "training_stats": {k: v for k, v in train_stats.items() if k != "label"},
        "oos_stats": {k: v for k, v in oos_stats.items() if k != "label"},
        "oos_wr_diff_pp": round(oos_wr_diff, 1),
        "monotonicity": mono,
        "per_day_analysis": day_ana,
        "monte_carlo": mc,
        "top15_sweep": [
            {k: v for k, v in r.items() if k != "label"} for r in all_results[:15]
        ],
        "verdict": verdict,
    }

    out_path = DIAG / "sbdr_results.json"
    with open(str(out_path), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
