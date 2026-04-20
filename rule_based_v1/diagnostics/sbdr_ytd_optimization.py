"""SBDR YTD 2026 Optimization.

Tests the SBDR (Sequential Burst Diffusion Reverter) strategy on YTD 2026
(Jan–Mar 2026, bearish/tariff-fear regime) with direction asymmetry analysis,
exit parameter sweeps, and entry enhancement sweeps.

Run:
    python rule_based_v1/diagnostics/sbdr_ytd_optimization.py
"""

from __future__ import annotations

import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
DIAG = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_mes_ytd() -> pd.DataFrame:
    """Load MES 1-min Jan–Feb 2026 (ETH UTC -> RTH Eastern)."""
    with pd.HDFStore(str(ROOT / "data/processed/jan_feb_2026_oos_test_1m.h5"), "r") as s:
        mes_janfeb = s["/bars_1min"].copy()
    mes_janfeb.index = pd.to_datetime(mes_janfeb.index, utc=True).tz_convert("US/Eastern")
    mes_janfeb = mes_janfeb.sort_index()
    mes_ytd = mes_janfeb[
        ((mes_janfeb.index.hour == 9) & (mes_janfeb.index.minute >= 30)) |
        ((mes_janfeb.index.hour > 9) & (mes_janfeb.index.hour < 16))
    ]
    return mes_ytd[["open", "high", "low", "close", "volume"]].copy()


def load_mnq_ytd() -> pd.DataFrame:
    """Load MNQ 1-min Jan–Mar 2026 (already RTH Eastern tz)."""
    with pd.HDFStore(str(ROOT / "data/processed/mnq_2026ytd_1min.h5"), "r") as s:
        mnq_ytd = s["/bars_1min"].copy()
    return mnq_ytd[["open", "high", "low", "close", "volume"]].copy()


# ---------------------------------------------------------------------------
# Signal Computation
# ---------------------------------------------------------------------------

def compute_signals(
    bars: pd.DataFrame,
    vol_window: int = 20,
    alpha: float = 0.3,
    lam: float = 0.2,
    horizons: list[int] | None = None,
    w_window: int = 5,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns: sigma_hat, Z_min, Z_max, Z_hstar, h_star, W_t, W_prev
    Indexed same as bars, NaN for warmup rows.
    """
    if horizons is None:
        horizons = [3, 5, 8]

    opens  = bars["open"].values.astype(float)
    highs  = bars["high"].values.astype(float)
    lows   = bars["low"].values.astype(float)
    closes = bars["close"].values.astype(float)
    n = len(closes)

    sigma_hat = np.full(n, np.nan)
    Z_min_arr = np.full(n, np.nan)
    Z_max_arr = np.full(n, np.nan)
    Z_hstar_arr = np.full(n, np.nan)
    h_star_arr = np.full(n, np.nan)
    W_arr = np.full(n, np.nan)

    warmup = vol_window + max(horizons) + 1

    for i in range(warmup, n):
        # --- volatility ---
        win = slice(i - vol_window, i)
        H = highs[win]; L = lows[win]; O = opens[win]; C = closes[win]
        u = H - np.maximum(O, C)
        l = np.minimum(O, C) - L
        s2_wick = np.mean((u**2 + l**2) / 2.0)
        s2_range = np.mean((H - L)**2) / (4 * np.log(2))
        s2 = alpha * s2_wick + (1 - alpha) * s2_range
        sig = np.sqrt(max(s2, 1e-8))
        sigma_hat[i] = sig

        # --- Z scores ---
        Z_vals = []
        for h in horizons:
            weights = np.array([np.exp(-lam * k) for k in range(h)])
            rets = np.diff(closes[i - h: i + 1])  # length h, most recent last
            rets_rev = rets[::-1]                   # most recent first
            num = np.sum(weights * rets_rev)
            denom = sig * np.sqrt(np.sum(weights**2))
            Z_vals.append(num / (denom + 1e-10))

        Z_min_arr[i] = min(Z_vals)
        Z_max_arr[i] = max(Z_vals)

        # h_star = horizon that produced most extreme Z
        abs_z = [abs(z) for z in Z_vals]
        best_idx = int(np.argmax(abs_z))
        Z_hstar_arr[i] = Z_vals[best_idx]
        h_star_arr[i] = horizons[best_idx]

        # --- wick asymmetry W_t ---
        ws = slice(max(0, i - w_window + 1), i + 1)
        Hw = highs[ws]; Lw = lows[ws]; Ow = opens[ws]; Cw = closes[ws]
        uw = Hw - np.maximum(Ow, Cw)
        lw = np.minimum(Ow, Cw) - Lw
        u2 = np.sum(uw**2); l2 = np.sum(lw**2)
        W_arr[i] = (u2 - l2) / (u2 + l2 + 1e-10)

    return pd.DataFrame({
        "sigma_hat": sigma_hat,
        "Z_min": Z_min_arr,
        "Z_max": Z_max_arr,
        "Z_hstar": Z_hstar_arr,
        "h_star": h_star_arr,
        "W_t": W_arr,
        "W_prev": np.concatenate([[np.nan], W_arr[:-1]]),
    }, index=bars.index)


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------

def run_sbdr(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    u_pct: int = 82,
    stop_mult: float = 0.6,
    target_mult: float = 0.8,
    time_stop_bars: int = 5,
    n_contracts: int = 1,
    point_value: float = 5.0,
    commission: float = 0.62,
    slippage_ticks: int = 1,
    tick_size: float = 0.25,
    only_long: bool = False,
    only_short: bool = False,
    trend_filter: bool = False,
    min_abs_w_change: float = 0.0,
    sustained_bars: int = 1,
    entry_start_time: tuple[int, int] = (9, 35),
    entry_end_time: tuple[int, int] = (15, 40),
    percentile_window: int = 60,
    u_floor: float = 1.5,
    daily_loss_limit: float = 500.0,
    max_daily_losses: int = 3,
) -> list[dict]:
    """Run SBDR on pre-computed signals. Returns list of trade dicts."""
    opens  = bars["open"].values.astype(float)
    highs  = bars["high"].values.astype(float)
    lows   = bars["low"].values.astype(float)
    closes = bars["close"].values.astype(float)
    timestamps = bars.index

    n = len(bars)
    tick_value = point_value * tick_size
    slippage = slippage_ticks * tick_size

    # Compute 20-bar EMA for trend filter
    ema20 = pd.Series(closes, index=bars.index).ewm(span=20, adjust=False).mean().values

    sig_sigma    = signals["sigma_hat"].values
    sig_Z_min    = signals["Z_min"].values
    sig_Z_max    = signals["Z_max"].values
    sig_Z_hstar  = signals["Z_hstar"].values
    sig_h_star   = signals["h_star"].values
    sig_W_t      = signals["W_t"].values

    trades: list[dict] = []

    # Rolling |Z_hstar| buffer for dynamic threshold u_t
    z_abs_hist: list[float] = []

    # Session state
    session_date = None
    daily_pnl: float = 0.0
    session_loss_count: int = 0

    # Sustained Z tracking: circular buffer of last sustained_bars values of abs(Z_hstar)
    z_abs_recent: list[float] = []

    # Position state
    in_trade: bool = False
    direction: int = 0
    entry_price: float = 0.0
    entry_bar_idx: int = 0
    stop_dist: float = 0.0
    target_dist: float = 0.0
    trade_entry_time = None
    trade_entry_z: float = 0.0
    trade_entry_w_change: float = 0.0
    trade_entry_sigma: float = 0.0
    trade_entry_h_star: int = 3

    for i in range(n):
        ts = timestamps[i]
        bar_date = ts.normalize()

        # Session reset
        if bar_date != session_date:
            if in_trade:
                # Force close at prior bar's close on new session
                exit_p = closes[i - 1] if i > 0 else closes[i]
                raw_pnl = direction * (exit_p - entry_price) * n_contracts * point_value
                comm = 2.0 * commission * n_contracts
                pnl = raw_pnl - comm
                trades.append({
                    "date": str(session_date)[:10],
                    "direction": "LONG" if direction == 1 else "SHORT",
                    "entry": entry_price,
                    "exit": exit_p,
                    "pnl": round(pnl, 2),
                    "reason": "session_end",
                    "Z_trigger": round(trade_entry_z, 4),
                    "h_star": trade_entry_h_star,
                    "W_change": round(trade_entry_w_change, 4),
                    "sigma_hat": round(trade_entry_sigma, 4),
                    "bars_held": i - 1 - entry_bar_idx,
                })
                in_trade = False

            session_date = bar_date
            daily_pnl = 0.0
            session_loss_count = 0
            z_abs_recent = []

        # Force flat at 15:55
        if in_trade:
            bar_t = (ts.hour, ts.minute)
            if bar_t >= (15, 55):
                exit_p = closes[i]
                raw_pnl = direction * (exit_p - entry_price) * n_contracts * point_value
                comm = 2.0 * commission * n_contracts
                pnl = raw_pnl - comm
                daily_pnl += pnl
                if pnl <= 0:
                    session_loss_count += 1
                trades.append({
                    "date": str(session_date)[:10],
                    "direction": "LONG" if direction == 1 else "SHORT",
                    "entry": entry_price,
                    "exit": exit_p,
                    "pnl": round(pnl, 2),
                    "reason": "force_flat",
                    "Z_trigger": round(trade_entry_z, 4),
                    "h_star": trade_entry_h_star,
                    "W_change": round(trade_entry_w_change, 4),
                    "sigma_hat": round(trade_entry_sigma, 4),
                    "bars_held": i - entry_bar_idx,
                })
                in_trade = False
                continue

        # Skip if signals not ready
        if np.isnan(sig_sigma[i]) or np.isnan(sig_Z_hstar[i]):
            continue

        # Update rolling Z abs history
        z_abs_now = abs(sig_Z_hstar[i])
        z_abs_hist.append(z_abs_now)
        if len(z_abs_hist) > percentile_window:
            z_abs_hist.pop(0)

        if len(z_abs_hist) < 5:
            continue

        u_t = max(np.percentile(z_abs_hist, u_pct), u_floor)

        # Update sustained bars buffer
        z_abs_recent.append(z_abs_now)
        if len(z_abs_recent) > sustained_bars:
            z_abs_recent.pop(0)

        w_t = sig_W_t[i]
        w_prev = sig_W_t[i - 1] if i > 0 and not np.isnan(sig_W_t[i - 1]) else w_t

        # ---- Exit check for open trade ----
        if in_trade:
            bars_held = i - entry_bar_idx
            exited = False
            reason = ""

            if direction == 1:  # long
                if lows[i] <= entry_price - stop_dist:
                    exit_p = entry_price - stop_dist
                    reason = "stop_loss"
                    exited = True
                elif highs[i] >= entry_price + target_dist:
                    exit_p = entry_price + target_dist
                    reason = "take_profit"
                    exited = True
                elif bars_held >= time_stop_bars:
                    exit_p = closes[i]
                    reason = "time_stop"
                    exited = True
            else:  # short
                if highs[i] >= entry_price + stop_dist:
                    exit_p = entry_price + stop_dist
                    reason = "stop_loss"
                    exited = True
                elif lows[i] <= entry_price - target_dist:
                    exit_p = entry_price - target_dist
                    reason = "take_profit"
                    exited = True
                elif bars_held >= time_stop_bars:
                    exit_p = closes[i]
                    reason = "time_stop"
                    exited = True

            if exited:
                raw_pnl = direction * (exit_p - entry_price) * n_contracts * point_value
                comm = 2.0 * commission * n_contracts
                pnl = raw_pnl - comm
                daily_pnl += pnl
                if pnl <= 0:
                    session_loss_count += 1
                trades.append({
                    "date": str(session_date)[:10],
                    "direction": "LONG" if direction == 1 else "SHORT",
                    "entry": entry_price,
                    "exit": exit_p,
                    "pnl": round(pnl, 2),
                    "reason": reason,
                    "Z_trigger": round(trade_entry_z, 4),
                    "h_star": trade_entry_h_star,
                    "W_change": round(trade_entry_w_change, 4),
                    "sigma_hat": round(trade_entry_sigma, 4),
                    "bars_held": bars_held,
                })
                in_trade = False
                continue

        # ---- Entry check ----
        bar_t = (ts.hour, ts.minute)
        time_ok = entry_start_time <= bar_t <= entry_end_time

        if (
            not in_trade
            and time_ok
            and session_loss_count < max_daily_losses
            and daily_pnl > -daily_loss_limit
        ):
            sigma_i = sig_sigma[i]
            h_star_i = int(sig_h_star[i])
            z_min_i  = sig_Z_min[i]
            z_max_i  = sig_Z_max[i]

            setup_range = highs[i] - lows[i]
            stop_dist_raw = stop_mult * setup_range
            stop_dist_calc = max(stop_dist_raw, 5.0 * tick_size)  # floor 5 ticks

            target_raw = target_mult * sigma_i * np.sqrt(h_star_i)
            target_dist_calc = max(target_raw, stop_dist_calc + 3.0 * tick_size)

            delta_w = abs(w_t - w_prev)

            # Sustained Z check: all recent bars must exceed u_t
            sustained_ok = (
                len(z_abs_recent) >= sustained_bars
                and all(z >= u_t for z in z_abs_recent[-sustained_bars:])
            )

            # Trend filter state
            above_ema = closes[i] > ema20[i]

            # Long entry: z_min < -u_t (extreme downward burst -> fade up)
            long_signal = z_min_i < -u_t and sustained_ok
            if not only_short and long_signal:
                if trend_filter and not above_ema:
                    long_signal = False
            if not only_short and not only_long:
                pass  # both allowed

            if only_long and z_max_i > u_t:
                pass  # skip short signals

            # ---- Long entry ----
            if (not only_short) and z_min_i < -u_t and sustained_ok:
                if not (trend_filter and not above_ema):
                    if delta_w >= min_abs_w_change:
                        direction = 1
                        entry_price = closes[i] + slippage
                        entry_bar_idx = i
                        trade_entry_time = ts
                        trade_entry_z = abs(z_min_i)
                        trade_entry_w_change = delta_w
                        trade_entry_sigma = sigma_i
                        trade_entry_h_star = h_star_i
                        stop_dist = stop_dist_calc
                        target_dist = target_dist_calc
                        in_trade = True

            # ---- Short entry (only if no long) ----
            elif (not only_long) and z_max_i > u_t and sustained_ok:
                if not (trend_filter and above_ema):
                    if delta_w >= min_abs_w_change:
                        direction = -1
                        entry_price = closes[i] - slippage
                        entry_bar_idx = i
                        trade_entry_time = ts
                        trade_entry_z = abs(z_max_i)
                        trade_entry_w_change = delta_w
                        trade_entry_sigma = sigma_i
                        trade_entry_h_star = h_star_i
                        stop_dist = stop_dist_calc
                        target_dist = target_dist_calc
                        in_trade = True

    return trades


# ---------------------------------------------------------------------------
# Summary Stats
# ---------------------------------------------------------------------------

def stats(trades: list[dict], label: str = "") -> dict:
    if not trades:
        return {
            "label": label, "N": 0, "n_days": 0, "n_per_day": 0,
            "WR": 0, "avg_win": 0, "avg_loss": 0, "pf": 0,
            "total_pnl": 0, "sharpe": 0, "max_dd": 0, "daily_pnl": {},
        }
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    daily: dict[str, float] = {}
    for t in trades:
        d = str(t["date"])[:10]
        daily[d] = daily.get(d, 0.0) + t["pnl"]

    dpnls = list(daily.values())
    active = [d for d in dpnls if d != 0]
    mean_d = np.mean(active) if active else 0.0
    std_d = np.std(active, ddof=1) if len(active) > 1 else 1e-9
    sharpe = mean_d / std_d * np.sqrt(252) if std_d > 1e-9 else 0.0

    eq = np.cumsum(pnls)
    pk = np.maximum.accumulate(eq)
    maxdd = float((eq - pk).min()) if len(eq) else 0.0
    n_days = len([d for d in dpnls if d != 0])

    return {
        "label": label,
        "N": len(pnls),
        "n_days": n_days,
        "n_per_day": round(len(pnls) / max(n_days, 1), 2),
        "WR": round(100 * len(wins) / max(len(pnls), 1), 1),
        "avg_win": round(np.mean(wins), 2) if wins else 0,
        "avg_loss": round(np.mean(losses), 2) if losses else 0,
        "pf": round(sum(wins) / abs(sum(losses)), 2) if losses else 999,
        "total_pnl": round(sum(pnls), 2),
        "sharpe": round(sharpe, 3),
        "max_dd": round(maxdd, 2),
        "daily_pnl": daily,
    }


def print_stats(s: dict) -> None:
    label = s.get("label", "")
    if label:
        print(f"  [{label}]")
    print(f"    N={s['N']}  n_days={s['n_days']}  n/day={s['n_per_day']}")
    print(f"    WR={s['WR']}%  avg_win=${s['avg_win']}  avg_loss=${s['avg_loss']}  PF={s['pf']}")
    print(f"    Total PnL=${s['total_pnl']}  Sharpe={s['sharpe']}  MaxDD=${s['max_dd']}")


# ---------------------------------------------------------------------------
# Monotonicity Validation
# ---------------------------------------------------------------------------

def monotonicity_analysis(trades: list[dict], label: str = "") -> dict:
    """Bucket by |Z_trigger| and W_change to validate core hypothesis."""
    if len(trades) < 10:
        return {}

    z_vals = [t["Z_trigger"] for t in trades]
    w_vals = [t["W_change"] for t in trades]

    z_pcts = np.percentile(z_vals, [33, 67])
    w_pcts = np.percentile(w_vals, [33, 67])

    def bucket_z(z: float) -> str:
        if z < z_pcts[0]: return "low_Z"
        if z < z_pcts[1]: return "mid_Z"
        return "high_Z"

    def bucket_w(w: float) -> str:
        if w < w_pcts[0]: return "low_W"
        if w < w_pcts[1]: return "mid_W"
        return "high_W"

    z_buckets: dict[str, list[float]] = {"low_Z": [], "mid_Z": [], "high_Z": []}
    w_buckets: dict[str, list[float]] = {"low_W": [], "mid_W": [], "high_W": []}

    for t in trades:
        z_buckets[bucket_z(t["Z_trigger"])].append(t["pnl"])
        w_buckets[bucket_w(t["W_change"])].append(t["pnl"])

    def bucket_stats(buckets: dict[str, list[float]]) -> dict[str, dict]:
        out = {}
        for name, pnls in buckets.items():
            if not pnls:
                out[name] = {"N": 0, "WR": 0, "avg_pnl": 0}
                continue
            wins = [p for p in pnls if p > 0]
            out[name] = {
                "N": len(pnls),
                "WR": round(100 * len(wins) / len(pnls), 1),
                "avg_pnl": round(np.mean(pnls), 2),
            }
        return out

    z_analysis = bucket_stats(z_buckets)
    w_analysis = bucket_stats(w_buckets)

    # Monotonicity check: WR should increase from low to high
    z_wrs = [z_analysis[k]["WR"] for k in ["low_Z", "mid_Z", "high_Z"]]
    w_wrs = [w_analysis[k]["WR"] for k in ["low_W", "mid_W", "high_W"]]
    z_monotone = z_wrs[0] <= z_wrs[1] <= z_wrs[2]
    w_monotone = w_wrs[0] <= w_wrs[1] <= w_wrs[2]

    return {
        "label": label,
        "Z_buckets": z_analysis,
        "W_buckets": w_analysis,
        "Z_monotone": z_monotone,
        "W_monotone": w_monotone,
        "Z_WRs": z_wrs,
        "W_WRs": w_wrs,
    }


def print_monotonicity(m: dict) -> None:
    if not m:
        print("  (insufficient trades for analysis)")
        return
    print(f"  Z-bucket monotonicity: {m['Z_monotone']}  WRs={m['Z_WRs']}")
    for k, v in m["Z_buckets"].items():
        print(f"    {k}: N={v['N']}  WR={v['WR']}%  avg_pnl=${v['avg_pnl']}")
    print(f"  W-bucket monotonicity: {m['W_monotone']}  WRs={m['W_WRs']}")
    for k, v in m["W_buckets"].items():
        print(f"    {k}: N={v['N']}  WR={v['WR']}%  avg_pnl=${v['avg_pnl']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("SBDR YTD 2026 OPTIMIZATION")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    mes_ytd = load_mes_ytd()
    mnq_ytd = load_mnq_ytd()
    print(f"  MES YTD (Jan-Feb 2026): {len(mes_ytd)} bars, "
          f"{mes_ytd.index[0].date()} to {mes_ytd.index[-1].date()}")
    print(f"  MNQ YTD (Jan-Mar 2026): {len(mnq_ytd)} bars, "
          f"{mnq_ytd.index[0].date()} to {mnq_ytd.index[-1].date()}")

    # Pre-compute signals (best training config: alpha=0.3, lam=0.2)
    print("\nComputing signals...")
    mes_sigs = compute_signals(mes_ytd, alpha=0.3, lam=0.2)
    mnq_sigs = compute_signals(mnq_ytd, alpha=0.3, lam=0.2)
    print("  Done.")

    # Best training config baseline parameters
    BASE_U_PCT    = 88
    BASE_STOP     = 0.6
    BASE_TARGET   = 0.8
    BASE_TSBAR    = 5

    all_results: dict = {}

    # -----------------------------------------------------------------------
    # SECTION 1: Base Run on YTD Data
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 1: BASE RUN ON YTD DATA (u_pct=88, alpha=0.3, lam=0.2)")
    print("=" * 70)

    mes_base_trades = run_sbdr(
        mes_ytd, mes_sigs,
        u_pct=BASE_U_PCT, stop_mult=BASE_STOP, target_mult=BASE_TARGET,
        time_stop_bars=BASE_TSBAR, point_value=5.0,
    )
    mes_base_stats = stats(mes_base_trades, label="MES Jan-Feb 2026 (base)")
    print("\nMES Jan-Feb 2026:")
    print_stats(mes_base_stats)

    mnq_base_trades = run_sbdr(
        mnq_ytd, mnq_sigs,
        u_pct=BASE_U_PCT, stop_mult=BASE_STOP, target_mult=BASE_TARGET,
        time_stop_bars=BASE_TSBAR, point_value=2.0, tick_size=0.25,
    )
    mnq_base_stats = stats(mnq_base_trades, label="MNQ Jan-Mar 2026 (base)")
    print("\nMNQ Jan-Mar 2026 (point_value=2.0):")
    print_stats(mnq_base_stats)

    all_results["section1_base"] = {
        "mes_ytd": mes_base_stats,
        "mnq_ytd": mnq_base_stats,
    }

    # -----------------------------------------------------------------------
    # SECTION 2: Direction Asymmetry
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 2: DIRECTION ASYMMETRY (MES YTD, u_pct=88)")
    print("=" * 70)

    long_trades = run_sbdr(
        mes_ytd, mes_sigs,
        u_pct=BASE_U_PCT, stop_mult=BASE_STOP, target_mult=BASE_TARGET,
        time_stop_bars=BASE_TSBAR, point_value=5.0, only_long=True,
    )
    short_trades = run_sbdr(
        mes_ytd, mes_sigs,
        u_pct=BASE_U_PCT, stop_mult=BASE_STOP, target_mult=BASE_TARGET,
        time_stop_bars=BASE_TSBAR, point_value=5.0, only_short=True,
    )

    long_stats  = stats(long_trades,  label="LONG only")
    short_stats = stats(short_trades, label="SHORT only")
    both_stats  = mes_base_stats

    print("\nBoth directions:")
    print_stats(both_stats)
    print("\nLONG only:")
    print_stats(long_stats)
    print("\nSHORT only:")
    print_stats(short_stats)

    all_results["section2_direction"] = {
        "both":  {k: v for k, v in both_stats.items()  if k != "daily_pnl"},
        "long":  {k: v for k, v in long_stats.items()  if k != "daily_pnl"},
        "short": {k: v for k, v in short_stats.items() if k != "daily_pnl"},
    }

    # -----------------------------------------------------------------------
    # SECTION 3: Exit Parameter Sweep
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 3: EXIT PARAMETER SWEEP (MES YTD, u_pct=88)")
    print("=" * 70)

    stop_mults   = [0.4, 0.5, 0.6, 0.7, 0.8]
    target_mults = [0.6, 0.8, 1.0, 1.2, 1.5]
    time_stops   = [3, 4, 5, 6]

    total_exit = len(stop_mults) * len(target_mults) * len(time_stops)
    print(f"  Running {total_exit} combos...")

    exit_results: list[dict] = []
    for sm, tm, ts_b in product(stop_mults, target_mults, time_stops):
        t = run_sbdr(
            mes_ytd, mes_sigs,
            u_pct=BASE_U_PCT, stop_mult=sm, target_mult=tm,
            time_stop_bars=ts_b, point_value=5.0,
        )
        s = stats(t)
        exit_results.append({
            "stop_mult": sm, "target_mult": tm, "time_stop_bars": ts_b,
            "N": s["N"], "WR": s["WR"], "sharpe": s["sharpe"],
            "total_pnl": s["total_pnl"], "max_dd": s["max_dd"],
        })

    exit_results.sort(key=lambda x: x["sharpe"], reverse=True)
    print("\nTop 10 exit combos by Sharpe:")
    print(f"  {'stop':>5} {'tgt':>5} {'tsb':>4}  {'N':>4}  {'WR':>5}  {'Sharpe':>7}  {'PnL':>8}  {'MaxDD':>8}")
    for r in exit_results[:10]:
        print(f"  {r['stop_mult']:>5.1f} {r['target_mult']:>5.1f} {r['time_stop_bars']:>4d}  "
              f"{r['N']:>4d}  {r['WR']:>5.1f}%  {r['sharpe']:>7.3f}  "
              f"${r['total_pnl']:>7.0f}  ${r['max_dd']:>7.0f}")

    best_exit = exit_results[0]
    BEST_STOP   = best_exit["stop_mult"]
    BEST_TARGET = best_exit["target_mult"]
    BEST_TSBAR  = best_exit["time_stop_bars"]
    print(f"\nBest exit: stop={BEST_STOP}, target={BEST_TARGET}, time_stop={BEST_TSBAR}")
    print(f"  Sharpe={best_exit['sharpe']}, N={best_exit['N']}, WR={best_exit['WR']}%, PnL=${best_exit['total_pnl']}")

    all_results["section3_exit_sweep"] = {
        "top10": exit_results[:10],
        "best": best_exit,
    }

    # -----------------------------------------------------------------------
    # SECTION 4: Entry Enhancement Sweep
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 4: ENTRY ENHANCEMENT SWEEP (best exit params)")
    print("=" * 70)

    # Baseline for comparison
    base_entry_trades = run_sbdr(
        mes_ytd, mes_sigs,
        u_pct=BASE_U_PCT, stop_mult=BEST_STOP, target_mult=BEST_TARGET,
        time_stop_bars=BEST_TSBAR, point_value=5.0,
    )
    base_entry_stats = stats(base_entry_trades, label="baseline")

    # --- 4a: W_t change threshold ---
    print("\n4a. W_t change threshold (min_abs_w_change):")
    w_change_vals = [0.0, 0.03, 0.05, 0.08, 0.12]
    w_change_results: list[dict] = []
    for wc in w_change_vals:
        t = run_sbdr(
            mes_ytd, mes_sigs,
            u_pct=BASE_U_PCT, stop_mult=BEST_STOP, target_mult=BEST_TARGET,
            time_stop_bars=BEST_TSBAR, point_value=5.0, min_abs_w_change=wc,
        )
        s = stats(t)
        w_change_results.append({"min_abs_w_change": wc, **{k: v for k, v in s.items() if k != "daily_pnl"}})
        print(f"  w_change={wc:.2f}: N={s['N']}  WR={s['WR']}%  Sharpe={s['sharpe']}  PnL=${s['total_pnl']}")
    all_results["section4a_w_change"] = w_change_results

    # --- 4b: Sustained burst ---
    print("\n4b. Sustained burst (sustained_bars):")
    sustained_results: list[dict] = []
    for sb in [1, 2, 3]:
        t = run_sbdr(
            mes_ytd, mes_sigs,
            u_pct=BASE_U_PCT, stop_mult=BEST_STOP, target_mult=BEST_TARGET,
            time_stop_bars=BEST_TSBAR, point_value=5.0, sustained_bars=sb,
        )
        s = stats(t)
        sustained_results.append({"sustained_bars": sb, **{k: v for k, v in s.items() if k != "daily_pnl"}})
        print(f"  sustained_bars={sb}: N={s['N']}  WR={s['WR']}%  Sharpe={s['sharpe']}  PnL=${s['total_pnl']}")
    all_results["section4b_sustained"] = sustained_results

    # --- 4c: Trend filter ---
    print("\n4c. Trend filter (trend_filter=True: with-trend only):")
    for tf_flag, lbl in [(False, "no trend filter"), (True, "with trend filter (EMA20)")]:
        t = run_sbdr(
            mes_ytd, mes_sigs,
            u_pct=BASE_U_PCT, stop_mult=BEST_STOP, target_mult=BEST_TARGET,
            time_stop_bars=BEST_TSBAR, point_value=5.0, trend_filter=tf_flag,
        )
        s = stats(t)
        print(f"  {lbl}: N={s['N']}  WR={s['WR']}%  Sharpe={s['sharpe']}  PnL=${s['total_pnl']}")
    trend_true_trades = run_sbdr(
        mes_ytd, mes_sigs,
        u_pct=BASE_U_PCT, stop_mult=BEST_STOP, target_mult=BEST_TARGET,
        time_stop_bars=BEST_TSBAR, point_value=5.0, trend_filter=True,
    )
    trend_stats = stats(trend_true_trades)
    all_results["section4c_trend"] = {
        "no_filter":   {k: v for k, v in base_entry_stats.items() if k != "daily_pnl"},
        "with_filter": {k: v for k, v in trend_stats.items() if k != "daily_pnl"},
    }

    # --- 4d: Time-of-day filter ---
    print("\n4d. Time-of-day filter:")
    time_windows = {
        "all_day":   ((9, 35), (15, 40)),
        "morning":   ((9, 35), (11, 30)),
        "midday":    ((11, 0), (14, 0)),
        "afternoon": ((13, 0), (15, 40)),
    }
    tod_results: list[dict] = []
    for window_name, (t_start, t_end) in time_windows.items():
        t = run_sbdr(
            mes_ytd, mes_sigs,
            u_pct=BASE_U_PCT, stop_mult=BEST_STOP, target_mult=BEST_TARGET,
            time_stop_bars=BEST_TSBAR, point_value=5.0,
            entry_start_time=t_start, entry_end_time=t_end,
        )
        s = stats(t)
        tod_results.append({"window": window_name, **{k: v for k, v in s.items() if k != "daily_pnl"}})
        print(f"  {window_name:12s}: N={s['N']}  WR={s['WR']}%  Sharpe={s['sharpe']}  PnL=${s['total_pnl']}")
    all_results["section4d_tod"] = tod_results

    # -----------------------------------------------------------------------
    # SECTION 5: Best Combined Config
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 5: BEST COMBINED CONFIG (Sections 3+4)")
    print("=" * 70)

    # Build candidates from Sections 4a/4b/4c/4d combined with best exit
    # Sweep the most promising entry enhancements together
    best_wc  = max(w_change_results, key=lambda x: x["sharpe"])["min_abs_w_change"]
    best_sb  = max(sustained_results, key=lambda x: x["sharpe"])["sustained_bars"]
    best_tf  = trend_stats["sharpe"] > base_entry_stats["sharpe"]
    best_tod = max(tod_results, key=lambda x: x["sharpe"])

    # Run a focused combined sweep
    print(f"\nBest individual enhancements:")
    print(f"  min_abs_w_change = {best_wc}")
    print(f"  sustained_bars   = {best_sb}")
    print(f"  trend_filter     = {best_tf}")
    print(f"  time_window      = {best_tod['window']}")

    best_tod_start, best_tod_end = time_windows[best_tod["window"]]

    combined_candidates: list[dict] = []
    # Sweep combinations of the top entry enhancements
    for wc in [0.0, best_wc]:
        for sb in [1, best_sb]:
            for tf in [False, best_tf] if best_tf else [False]:
                for win_name, (t_start, t_end) in [("all_day", ((9,35),(15,40))), (best_tod["window"], (best_tod_start, best_tod_end))]:
                    if win_name == "all_day" and best_tod["window"] == "all_day":
                        if (wc, sb, tf) != (0.0, 1, False):  # avoid dup of baseline
                            pass
                    t = run_sbdr(
                        mes_ytd, mes_sigs,
                        u_pct=BASE_U_PCT, stop_mult=BEST_STOP, target_mult=BEST_TARGET,
                        time_stop_bars=BEST_TSBAR, point_value=5.0,
                        min_abs_w_change=wc, sustained_bars=sb,
                        trend_filter=tf,
                        entry_start_time=t_start, entry_end_time=t_end,
                    )
                    s = stats(t)
                    combined_candidates.append({
                        "wc": wc, "sb": sb, "tf": tf, "window": win_name,
                        "N": s["N"], "WR": s["WR"], "sharpe": s["sharpe"],
                        "total_pnl": s["total_pnl"], "max_dd": s["max_dd"],
                    })

    # Filter N >= 30 and sort by Sharpe
    valid_candidates = [c for c in combined_candidates if c["N"] >= 30]
    if not valid_candidates:
        valid_candidates = combined_candidates  # relax if needed
    valid_candidates.sort(key=lambda x: x["sharpe"], reverse=True)
    best_combined = valid_candidates[0]

    print(f"\nBest combined config (N>={30}, sorted by Sharpe):")
    print(f"  wc={best_combined['wc']}, sb={best_combined['sb']}, "
          f"tf={best_combined['tf']}, window={best_combined['window']}")
    print(f"  N={best_combined['N']}, WR={best_combined['WR']}%, "
          f"Sharpe={best_combined['sharpe']}, PnL=${best_combined['total_pnl']}, "
          f"MaxDD=${best_combined['max_dd']}")

    best_tod_s, best_tod_e = time_windows[best_combined["window"]]

    # Run best combined on MES
    best_mes_trades = run_sbdr(
        mes_ytd, mes_sigs,
        u_pct=BASE_U_PCT, stop_mult=BEST_STOP, target_mult=BEST_TARGET,
        time_stop_bars=BEST_TSBAR, point_value=5.0,
        min_abs_w_change=best_combined["wc"],
        sustained_bars=best_combined["sb"],
        trend_filter=best_combined["tf"],
        entry_start_time=best_tod_s, entry_end_time=best_tod_e,
    )
    best_mes_stats = stats(best_mes_trades, label="MES Jan-Feb 2026 (best combined)")
    print("\nMES Jan-Feb 2026 (best combined config):")
    print_stats(best_mes_stats)

    # Run best combined on MNQ
    best_mnq_trades = run_sbdr(
        mnq_ytd, mnq_sigs,
        u_pct=BASE_U_PCT, stop_mult=BEST_STOP, target_mult=BEST_TARGET,
        time_stop_bars=BEST_TSBAR, point_value=2.0, tick_size=0.25,
        min_abs_w_change=best_combined["wc"],
        sustained_bars=best_combined["sb"],
        trend_filter=best_combined["tf"],
        entry_start_time=best_tod_s, entry_end_time=best_tod_e,
    )
    best_mnq_stats = stats(best_mnq_trades, label="MNQ Jan-Mar 2026 (best combined)")
    print("\nMNQ Jan-Mar 2026 (best combined config, point_value=2.0):")
    print_stats(best_mnq_stats)

    # Monotonicity validation
    print("\nMonotonicity validation for best combined config:")
    print("  MES:")
    mes_mono = monotonicity_analysis(best_mes_trades, label="MES best combined")
    print_monotonicity(mes_mono)
    print("  MNQ:")
    mnq_mono = monotonicity_analysis(best_mnq_trades, label="MNQ best combined")
    print_monotonicity(mnq_mono)

    all_results["section5_best_combined"] = {
        "config": {
            "u_pct": BASE_U_PCT,
            "stop_mult": BEST_STOP,
            "target_mult": BEST_TARGET,
            "time_stop_bars": BEST_TSBAR,
            "min_abs_w_change": best_combined["wc"],
            "sustained_bars": best_combined["sb"],
            "trend_filter": best_combined["tf"],
            "time_window": best_combined["window"],
        },
        "mes_stats": {k: v for k, v in best_mes_stats.items() if k != "daily_pnl"},
        "mnq_stats": {k: v for k, v in best_mnq_stats.items() if k != "daily_pnl"},
        "mes_monotonicity": mes_mono,
        "mnq_monotonicity": mnq_mono,
    }

    # -----------------------------------------------------------------------
    # SECTION 6: Comparison vs Training
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 6: COMPARISON VS TRAINING (Jun–Dec 2025)")
    print("=" * 70)

    # Training period best config stats (from sbdr_backtest.py knowledge):
    # u_pct=88, alpha=0.3, lam=0.2 on mes_1m_bars_cache.h5
    # We re-run here for an apples-to-apples comparison
    try:
        with pd.HDFStore(str(ROOT / "data/processed/mes_1m_bars_cache.h5"), "r") as s:
            keys = list(s.keys())
            raw_train = s[keys[0]]

        # Handle possible timestamp column
        if "timestamp" in raw_train.columns:
            raw_train = raw_train.set_index("timestamp")
        raw_train.index = pd.to_datetime(raw_train.index, utc=True).tz_convert("US/Eastern")
        raw_train = raw_train.sort_index()
        train_bars = raw_train[
            (raw_train.index.hour > 9) | ((raw_train.index.hour == 9) & (raw_train.index.minute >= 30))
        ]
        train_bars = train_bars[train_bars.index.hour < 16]
        train_bars = train_bars[["open", "high", "low", "close", "volume"]].copy()

        print(f"\nTraining data: {train_bars.index[0].date()} to {train_bars.index[-1].date()}"
              f" ({len(train_bars)} bars)")

        train_sigs = compute_signals(train_bars, alpha=0.3, lam=0.2)
        train_base_trades = run_sbdr(
            train_bars, train_sigs,
            u_pct=BASE_U_PCT, stop_mult=BASE_STOP, target_mult=BASE_TARGET,
            time_stop_bars=BASE_TSBAR, point_value=5.0,
        )
        train_base_stats = stats(train_base_trades, label="Training Jun-Dec 2025 (base config)")

        # Best combined on training
        train_best_trades = run_sbdr(
            train_bars, train_sigs,
            u_pct=BASE_U_PCT, stop_mult=BEST_STOP, target_mult=BEST_TARGET,
            time_stop_bars=BEST_TSBAR, point_value=5.0,
            min_abs_w_change=best_combined["wc"],
            sustained_bars=best_combined["sb"],
            trend_filter=best_combined["tf"],
            entry_start_time=best_tod_s, entry_end_time=best_tod_e,
        )
        train_best_stats = stats(train_best_trades, label="Training Jun-Dec 2025 (best combined)")

        print("\nSide-by-side comparison (base config u_pct=88):")
        print("-" * 50)
        print("Training (Jun-Dec 2025):")
        print_stats(train_base_stats)
        print("YTD 2026 (Jan-Feb 2026, MES):")
        print_stats(mes_base_stats)

        print("\nSide-by-side comparison (best combined config from YTD optimization):")
        print("-" * 50)
        print("Training (Jun-Dec 2025):")
        print_stats(train_best_stats)
        print("YTD 2026 (Jan-Feb 2026, MES):")
        print_stats(best_mes_stats)

        train_sharpe_diff = best_mes_stats["sharpe"] - train_base_stats["sharpe"]
        same_config_note = (
            "Config transfers well across regimes (Sharpe improved in 2026 bear)"
            if train_sharpe_diff > 0
            else "Config requires regime-specific tuning (2026 bear regime underperforms training)"
        )
        print(f"\nRegime analysis: {same_config_note}")
        print(f"  Training Sharpe (base): {train_base_stats['sharpe']}")
        print(f"  YTD 2026 Sharpe (base): {mes_base_stats['sharpe']}")
        print(f"  Delta: {train_sharpe_diff:+.3f}")

        all_results["section6_vs_training"] = {
            "training_base":       {k: v for k, v in train_base_stats.items() if k != "daily_pnl"},
            "training_best":       {k: v for k, v in train_best_stats.items() if k != "daily_pnl"},
            "ytd2026_mes_base":    {k: v for k, v in mes_base_stats.items() if k != "daily_pnl"},
            "ytd2026_mes_best":    {k: v for k, v in best_mes_stats.items() if k != "daily_pnl"},
            "sharpe_delta_base":   round(train_sharpe_diff, 3),
            "regime_note":         same_config_note,
        }

    except Exception as e:
        print(f"\n  [Warning] Could not load training data: {e}")
        print("  Skipping training comparison.")
        all_results["section6_vs_training"] = {"error": str(e)}

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    out_path = DIAG / "sbdr_ytd_results.json"

    def _serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(v) for v in obj]
        return obj

    with open(str(out_path), "w") as f:
        json.dump(_serialize(all_results), f, indent=2)

    print("\n" + "=" * 70)
    print(f"Results saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
