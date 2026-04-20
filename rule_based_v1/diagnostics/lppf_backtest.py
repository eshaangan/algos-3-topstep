"""
LPPF (Latent Pressure Propagator Filter) Backtest
State-space Kalman filter that tracks efficient price + transitory pressure
to generate mean-reversion signals on 1-min MES/MNQ bars.
"""
import json
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

with pd.HDFStore(str(ROOT / "data/processed/mes_1m_bars_cache.h5"), "r") as s:
    df_train_raw = s["/bars_1m"].set_index("timestamp")
df_train_raw.index = pd.to_datetime(df_train_raw.index, utc=True).tz_convert("US/Eastern")
df_train_raw = df_train_raw.sort_index()

with pd.HDFStore(str(ROOT / "data/processed/jan_feb_2026_oos_test_1m.h5"), "r") as s:
    df_oos_raw = s["/bars_1min"].copy()
df_oos_raw.index = pd.to_datetime(df_oos_raw.index, utc=True).tz_convert("US/Eastern")
df_oos_raw = df_oos_raw.sort_index()

with pd.HDFStore(str(ROOT / "data/processed/mnq_2026ytd_1min.h5"), "r") as s:
    df_mnq_raw = s["/bars_1min"].copy()
# MNQ index name may be ts_event; normalise
df_mnq_raw.index.name = "timestamp"


def rth(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        ((df.index.hour == 9) & (df.index.minute >= 30))
        | ((df.index.hour > 9) & (df.index.hour < 16))
    )
    return df[mask].copy()


train = rth(df_train_raw)
oos = rth(df_oos_raw)
mnq = df_mnq_raw.copy()  # already RTH

print(f"Train bars : {len(train):,}  ({train.index[0].date()} – {train.index[-1].date()})")
print(f"OOS bars   : {len(oos):,}  ({oos.index[0].date()} – {oos.index[-1].date()})")
print(f"MNQ bars   : {len(mnq):,}  ({mnq.index[0].date()} – {mnq.index[-1].date()})")


# ---------------------------------------------------------------------------
# Pre-compute bar-level signals (vectorised)
# ---------------------------------------------------------------------------

def precompute(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars[["open", "high", "low", "close"]].copy()
    df.columns = ["O", "H", "L", "C"]

    rng = df["H"] - df["L"]

    # rolling vol proxies
    df["sigma_bar"] = rng.rolling(20, min_periods=1).mean()
    df["med60"] = rng.rolling(60, min_periods=1).median()

    # signed pressure proxy q_t
    rng_safe = rng + 1e-6
    CLV = 2 * (df["C"] - df["L"]) / rng_safe - 1
    Body = (df["C"] - df["O"]) / rng_safe
    upper_wick = df["H"] - df[["O", "C"]].max(axis=1)
    lower_wick = df[["O", "C"]].min(axis=1) - df["L"]
    WickSkew = (upper_wick - lower_wick) / rng_safe
    size_scale = np.sqrt(rng / (df["med60"] + 1e-6))
    raw_q = 1.2 * CLV + 0.8 * Body - 0.4 * WickSkew
    df["q"] = np.tanh(raw_q) * size_scale

    # observations
    df["y1"] = df["C"]
    df["y2"] = (df["H"] + df["L"] + 2 * df["C"]) / 4

    # wick variance for R matrix
    wick_sq = (upper_wick ** 2 + lower_wick ** 2) / 2
    df["wick_var"] = wick_sq.rolling(20, min_periods=1).mean()

    return df


train_pre = precompute(train)
oos_pre = precompute(oos)
mnq_pre = precompute(mnq)


# ---------------------------------------------------------------------------
# Kalman update (single bar)
# ---------------------------------------------------------------------------

def kalman_update(
    x_hat: np.ndarray,
    P: np.ndarray,
    phi: float,
    q_t: float,
    y_obs: np.ndarray,
    sigma_bar: float,
    wick_var: float,
):
    F = np.array([[1.0, 0.0], [0.0, phi]])
    B = np.array([0.0, 1.0])
    Q = np.diag([(0.15 * sigma_bar) ** 2, (0.60 * sigma_bar) ** 2])
    H_obs = np.array([[1.0, 1.0], [1.0, 0.5]])
    wick_var_safe = max(wick_var, (0.2 * sigma_bar) ** 2)
    R = np.diag(
        [
            max(wick_var_safe * 1.0, (0.5 * sigma_bar) ** 2),
            max(wick_var_safe * 0.4, (0.25 * sigma_bar) ** 2),
        ]
    )

    # Predict
    x_pred = F @ x_hat + B * q_t
    P_pred = F @ P @ F.T + Q

    # Update
    y_pred = H_obs @ x_pred
    S = H_obs @ P_pred @ H_obs.T + R
    K = P_pred @ H_obs.T @ np.linalg.inv(S)
    innov = y_obs - y_pred
    x_new = x_pred + K @ innov
    P_new = (np.eye(2) - K @ H_obs) @ P_pred

    # Pressure z-score
    p_var = max(P_new[1, 1], 1e-8)
    Z_p = x_new[1] / np.sqrt(p_var)

    return x_new, P_new, float(Z_p), float(x_new[1])


# ---------------------------------------------------------------------------
# Main backtest function
# ---------------------------------------------------------------------------

def run_lppf(
    bars: pd.DataFrame,
    phi: float = 0.65,
    z_star: float = 1.5,
    sl_k: float = 1.5,
    tp_frac: float = 0.8,
    time_stop_bars: int = 6,
    n_contracts: int = 1,
    point_value: float = 5.0,
    commission: float = 0.62,
    slippage_ticks: int = 1,
    tick_size: float = 0.25,
    decay_bars: int = 1,
    daily_loss_floor: float = -400.0,
    max_session_losses: int = 3,
) -> list:
    """Run LPPF backtest on pre-computed bar DataFrame."""

    slippage = slippage_ticks * tick_size
    cost_per_rt = 2 * commission  # round-trip commission

    trades = []

    # group by session (date)
    dates = sorted(set(bars.index.date))

    # global warmup: skip first 25 bars total
    global_bar_count = 0

    for day in dates:
        day_mask = bars.index.date == day
        day_bars = bars[day_mask]

        if len(day_bars) < 10:
            continue

        # session state
        session_losses = 0
        session_pnl = 0.0

        # Kalman state initialised at open
        open_price = float(day_bars["O"].iloc[0])
        sb0 = float(day_bars["sigma_bar"].iloc[0])
        x_hat = np.array([open_price, 0.0])
        P = np.diag([sb0 ** 2 * 4, sb0 ** 2 * 2])

        # per-session bar index for warmup
        session_bar = 0

        # position tracking
        in_trade = False
        direction = 0
        entry_price = 0.0
        entry_bar_idx = 0
        p_entry = 0.0
        P_entry_pp = 0.0
        Z_entry = 0.0
        D_entry = 0.0
        sigma_at_entry = 0.0
        bars_held = 0

        # decay history
        p_history = [0.0]  # seed

        for i, (ts, row) in enumerate(day_bars.iterrows()):
            global_bar_count += 1
            session_bar += 1

            sigma_bar = float(row["sigma_bar"])
            wick_var = float(row["wick_var"])
            q_t = float(row["q"])
            y_obs = np.array([row["y1"], row["y2"]])
            H = float(row["H"])
            L = float(row["L"])
            C = float(row["C"])

            x_hat, P, Z_p, p_filtered = kalman_update(
                x_hat, P, phi, q_t, y_obs, sigma_bar, wick_var
            )

            D_t = p_filtered - p_history[-1]
            p_history.append(p_filtered)

            hour = ts.hour
            minute = ts.minute

            is_session_close = (hour == 15 and minute >= 55)

            # ------- EXIT CHECK -------
            if in_trade:
                bars_held += 1
                exit_price = None
                exit_reason = None

                sl_dist = sl_k * np.sqrt(max(P_entry_pp, 1e-8))
                tp_level_abs = tp_frac * abs(p_entry)

                if direction == 1:  # LONG
                    # SL
                    if L <= entry_price - sl_dist:
                        exit_price = entry_price - sl_dist
                        exit_reason = "SL"
                    # TP: pressure reverts toward 0 or beyond tp_frac
                    elif p_filtered >= 0 or p_filtered >= -tp_level_abs:
                        exit_price = C - slippage
                        exit_reason = "TP"
                    # time stop
                    elif bars_held >= time_stop_bars:
                        exit_price = C
                        exit_reason = "time_stop"
                    # session close
                    elif is_session_close:
                        exit_price = C
                        exit_reason = "session_close"
                else:  # SHORT
                    # SL
                    if H >= entry_price + sl_dist:
                        exit_price = entry_price + sl_dist
                        exit_reason = "SL"
                    # TP
                    elif p_filtered <= 0 or p_filtered <= tp_level_abs:
                        exit_price = C + slippage
                        exit_reason = "TP"
                    # time stop
                    elif bars_held >= time_stop_bars:
                        exit_price = C
                        exit_reason = "time_stop"
                    # session close
                    elif is_session_close:
                        exit_price = C
                        exit_reason = "session_close"

                if exit_price is not None:
                    raw_pnl = direction * (exit_price - entry_price) * n_contracts * point_value
                    pnl = raw_pnl - cost_per_rt * n_contracts

                    trades.append(
                        {
                            "date": str(day),
                            "direction": direction,
                            "entry": entry_price,
                            "exit": exit_price,
                            "pnl": pnl,
                            "reason": exit_reason,
                            "Z_entry": Z_entry,
                            "p_entry": p_entry,
                            "P_entry_pp": P_entry_pp,
                            "D_entry": D_entry,
                            "bars_held": bars_held,
                            "sigma_at_entry": sigma_at_entry,
                        }
                    )

                    session_pnl += pnl
                    if pnl < 0:
                        session_losses += 1

                    in_trade = False
                    direction = 0
                    bars_held = 0

            # ------- ENTRY CHECK -------
            if not in_trade:
                # warmup: skip first 25 bars globally AND first 25 bars of session
                if global_bar_count <= 25 or session_bar <= 25:
                    continue

                # session guards
                if hour < 9 or (hour == 9 and minute < 35):
                    continue
                if hour > 15 or (hour == 15 and minute > 40):
                    continue

                # session loss circuit breaker
                if session_losses >= max_session_losses:
                    continue

                # daily loss floor
                if session_pnl <= daily_loss_floor:
                    continue

                # compute decay signal over last decay_bars bars
                if len(p_history) < decay_bars + 1:
                    continue

                recent_D = [
                    p_history[-k] - p_history[-k - 1]
                    for k in range(1, decay_bars + 1)
                ]

                long_signal = (Z_p < -z_star) and all(d > 0 for d in recent_D)
                short_signal = (Z_p > z_star) and all(d < 0 for d in recent_D)

                if long_signal:
                    entry_price = C + slippage
                    direction = 1
                elif short_signal:
                    entry_price = C - slippage
                    direction = -1
                else:
                    continue

                in_trade = True
                bars_held = 0
                p_entry = p_filtered
                P_entry_pp = P[1, 1]
                Z_entry = Z_p
                D_entry = D_t
                sigma_at_entry = sigma_bar

        # Force flat at EOD if still in trade
        if in_trade and len(day_bars) > 0:
            exit_price = float(day_bars["C"].iloc[-1])
            raw_pnl = direction * (exit_price - entry_price) * n_contracts * point_value
            pnl = raw_pnl - cost_per_rt * n_contracts
            trades.append(
                {
                    "date": str(day),
                    "direction": direction,
                    "entry": entry_price,
                    "exit": exit_price,
                    "pnl": pnl,
                    "reason": "eod_flat",
                    "Z_entry": Z_entry,
                    "p_entry": p_entry,
                    "P_entry_pp": P_entry_pp,
                    "D_entry": D_entry,
                    "bars_held": bars_held,
                    "sigma_at_entry": sigma_at_entry,
                }
            )
            in_trade = False

    return trades


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def compute_stats(trades: list, label: str = "", trading_days: int = None) -> dict:
    if not trades:
        return {"label": label, "N": 0}

    df = pd.DataFrame(trades)
    N = len(df)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    WR = len(wins) / N
    avg_win = wins["pnl"].mean() if len(wins) else 0.0
    avg_loss = losses["pnl"].mean() if len(losses) else 0.0
    total_pnl = df["pnl"].sum()

    # daily PnL series
    daily = df.groupby("date")["pnl"].sum()
    if trading_days is None:
        trading_days = max(len(daily), 1)

    n_per_day = N / trading_days

    # Sharpe (annualised daily)
    if len(daily) >= 2 and daily.std() > 0:
        sharpe = (daily.mean() / daily.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown (cumulative PnL)
    cum = df["pnl"].cumsum()
    roll_max = cum.cummax()
    dd = (cum - roll_max).min()

    return {
        "label": label,
        "N": N,
        "N_per_day": round(n_per_day, 2),
        "WR": round(WR, 4),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "total_pnl": round(total_pnl, 2),
        "sharpe": round(sharpe, 3),
        "max_dd": round(dd, 2),
        "trading_days": trading_days,
    }


def print_stats(stats: dict):
    if stats["N"] == 0:
        print(f"  {stats['label']}: NO TRADES")
        return
    print(
        f"  {stats['label']:30s} | N={stats['N']:4d} ({stats['N_per_day']:.2f}/day)"
        f" | WR={stats['WR']:.1%} | AvgW=${stats['avg_win']:.0f}"
        f" | AvgL=${stats['avg_loss']:.0f} | PnL=${stats['total_pnl']:,.0f}"
        f" | Sharpe={stats['sharpe']:.2f} | MaxDD=${stats['max_dd']:,.0f}"
    )


def count_trading_days(bars: pd.DataFrame) -> int:
    return len(set(bars.index.date))


# ---------------------------------------------------------------------------
# Section 1: Grid Sweep
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 1: GRID SWEEP (training Jun-Dec 2025)")
print("=" * 80)

phi_grid = [0.35, 0.50, 0.65, 0.80]
z_star_grid = [1.0, 1.25, 1.5, 1.75]
sl_k_grid = [1.0, 1.5, 2.0]
decay_bars_grid = [1, 2]

total_combos = len(phi_grid) * len(z_star_grid) * len(sl_k_grid) * len(decay_bars_grid)
print(f"Running {total_combos} combinations...")

train_days = count_trading_days(train_pre)
sweep_results = []

for phi, z_star, sl_k, decay_bars in product(phi_grid, z_star_grid, sl_k_grid, decay_bars_grid):
    trades = run_lppf(
        train_pre,
        phi=phi,
        z_star=z_star,
        sl_k=sl_k,
        decay_bars=decay_bars,
        n_contracts=1,
        point_value=5.0,
    )
    st = compute_stats(trades, trading_days=train_days)
    st.update({"phi": phi, "z_star": z_star, "sl_k": sl_k, "decay_bars": decay_bars})
    sweep_results.append(st)

sweep_df = pd.DataFrame(sweep_results)
sweep_df = sweep_df.sort_values("sharpe", ascending=False)

print("\nTop 15 by Sharpe:")
print(
    f"{'phi':>5} {'z*':>6} {'sl_k':>6} {'decay':>6} | "
    f"{'N':>5} {'N/day':>6} {'WR':>6} {'AvgW':>7} {'AvgL':>7} "
    f"{'PnL':>9} {'Sharpe':>7} {'MaxDD':>9}"
)
print("-" * 100)
for _, r in sweep_df.head(15).iterrows():
    print(
        f"{r['phi']:>5.2f} {r['z_star']:>6.2f} {r['sl_k']:>6.2f} {int(r['decay_bars']):>6d} | "
        f"{int(r['N']):>5d} {r['N_per_day']:>6.2f} {r['WR']:>6.1%} "
        f"{r['avg_win']:>7.0f} {r['avg_loss']:>7.0f} "
        f"{r['total_pnl']:>9,.0f} {r['sharpe']:>7.3f} {r['max_dd']:>9,.0f}"
    )

# Best config: highest Sharpe with N >= 50
eligible = sweep_df[sweep_df["N"] >= 50]
if eligible.empty:
    eligible = sweep_df
best_row = eligible.iloc[0]
best_phi = float(best_row["phi"])
best_z_star = float(best_row["z_star"])
best_sl_k = float(best_row["sl_k"])
best_decay = int(best_row["decay_bars"])

print(
    f"\nBest config: phi={best_phi}, z*={best_z_star}, sl_k={best_sl_k}, decay_bars={best_decay}"
)
print(
    f"  N={int(best_row['N'])}, Sharpe={best_row['sharpe']:.3f}, "
    f"WR={best_row['WR']:.1%}, PnL=${best_row['total_pnl']:,.0f}"
)

# Run best config and get full trade list
best_trades_train = run_lppf(
    train_pre,
    phi=best_phi,
    z_star=best_z_star,
    sl_k=best_sl_k,
    decay_bars=best_decay,
    n_contracts=1,
    point_value=5.0,
)
best_stats_train = compute_stats(best_trades_train, label="Train best", trading_days=train_days)

# ---------------------------------------------------------------------------
# Section 2: Monotonicity Tests
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 2: MONOTONICITY TESTS (training, best config)")
print("=" * 80)

mono_fail_count = 0

def tercile_stats(df_trades: pd.DataFrame, col: str, label: str):
    vals = df_trades[col].abs()
    p33 = vals.quantile(0.333)
    p66 = vals.quantile(0.666)
    buckets = pd.cut(vals, bins=[-np.inf, p33, p66, np.inf], labels=["B1", "B2", "B3"])
    results = []
    for b in ["B1", "B2", "B3"]:
        sub = df_trades[buckets == b]
        if len(sub) == 0:
            results.append({"bucket": b, "N": 0, "WR": np.nan, "avg_pnl": np.nan})
        else:
            WR = (sub["pnl"] > 0).mean()
            avg_pnl = sub["pnl"].mean()
            results.append({"bucket": b, "N": len(sub), "WR": WR, "avg_pnl": avg_pnl})
    return results


if best_trades_train:
    df_bt = pd.DataFrame(best_trades_train)

    # Z monotonicity
    z_res = tercile_stats(df_bt, "Z_entry", "|Z_entry|")
    print(f"\nZ-entry monotonicity (|Z_entry| buckets):")
    z_wrs = []
    for r in z_res:
        print(f"  {r['bucket']}: N={r['N']}, WR={r['WR']:.1%}, avg_pnl=${r['avg_pnl']:.0f}")
        z_wrs.append(r["WR"])
    z_mono = (not np.isnan(z_wrs[1]) and not np.isnan(z_wrs[2]) and
              z_wrs[0] <= z_wrs[1] <= z_wrs[2] + 0.02)  # allow tiny tolerance
    print(f"  Z monotonicity: {'PASS' if z_mono else 'FAIL'}")
    if not z_mono:
        mono_fail_count += 1

    # Decay monotonicity
    d_res = tercile_stats(df_bt, "D_entry", "|D_entry|")
    print(f"\nDecay monotonicity (|D_entry| buckets):")
    d_wrs = []
    for r in d_res:
        print(f"  {r['bucket']}: N={r['N']}, WR={r['WR']:.1%}, avg_pnl=${r['avg_pnl']:.0f}")
        d_wrs.append(r["WR"])
    d_mono = (not np.isnan(d_wrs[1]) and not np.isnan(d_wrs[2]) and
              d_wrs[0] <= d_wrs[1] <= d_wrs[2] + 0.02)
    print(f"  Decay monotonicity: {'PASS' if d_mono else 'FAIL'}")
    if not d_mono:
        mono_fail_count += 1

    # p_filtered monotonicity
    p_res = tercile_stats(df_bt, "p_entry", "|p_entry|")
    print(f"\np_entry monotonicity (|p_entry| buckets):")
    p_wrs = []
    for r in p_res:
        print(f"  {r['bucket']}: N={r['N']}, WR={r['WR']:.1%}, avg_pnl=${r['avg_pnl']:.0f}")
        p_wrs.append(r["WR"])
    p_mono = (not np.isnan(p_wrs[1]) and not np.isnan(p_wrs[2]) and
              p_wrs[0] <= p_wrs[1] <= p_wrs[2] + 0.02)
    print(f"  p_entry monotonicity: {'PASS' if p_mono else 'FAIL'}")
    if not p_mono:
        mono_fail_count += 1

    if mono_fail_count == 3:
        print("\n*** KILL: ALL THREE MONOTONICITIES FAILED ***")
        # Still continue to show full picture
else:
    print("No trades to assess monotonicity.")
    mono_fail_count = 3

# ---------------------------------------------------------------------------
# Section 3: Direction Split
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 3: DIRECTION SPLIT (training, best config)")
print("=" * 80)

if best_trades_train:
    df_bt = pd.DataFrame(best_trades_train)
    long_trades = [t for t in best_trades_train if t["direction"] == 1]
    short_trades = [t for t in best_trades_train if t["direction"] == -1]
    print_stats(compute_stats(long_trades, label="LONG only", trading_days=train_days))
    print_stats(compute_stats(short_trades, label="SHORT only", trading_days=train_days))

# ---------------------------------------------------------------------------
# Section 4: OOS Validation (MES Jan-Feb 2026)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 4: OOS VALIDATION (MES Jan-Feb 2026)")
print("=" * 80)

oos_days = count_trading_days(oos_pre)
oos_trades = run_lppf(
    oos_pre,
    phi=best_phi,
    z_star=best_z_star,
    sl_k=best_sl_k,
    decay_bars=best_decay,
    n_contracts=1,
    point_value=5.0,
)
oos_stats = compute_stats(oos_trades, label="OOS MES Jan-Feb 2026", trading_days=oos_days)
print_stats(oos_stats)

train_wr = best_stats_train.get("WR", 0.0)
oos_wr = oos_stats.get("WR", 0.0)
wr_gap = abs(train_wr - oos_wr)
print(
    f"\n  Train WR={train_wr:.1%}, OOS WR={oos_wr:.1%}, gap={wr_gap:.1%} "
    f"{'(GOOD <=8pp)' if wr_gap <= 0.08 else '(WARN >8pp)'}"
)

# ---------------------------------------------------------------------------
# Section 5: MNQ Transfer (Jan-Mar 2026)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 5: MNQ TRANSFER (Jan-Mar 2026, point_value=2.0)")
print("=" * 80)

mnq_days = count_trading_days(mnq_pre)
mnq_trades = run_lppf(
    mnq_pre,
    phi=best_phi,
    z_star=best_z_star,
    sl_k=best_sl_k,
    decay_bars=best_decay,
    n_contracts=1,
    point_value=2.0,
    tick_size=0.25,
    daily_loss_floor=-400.0,
)
mnq_stats = compute_stats(mnq_trades, label="MNQ Jan-Mar 2026", trading_days=mnq_days)
print_stats(mnq_stats)
print(f"  MNQ WR={mnq_stats.get('WR', 0):.1%} vs MES train WR={train_wr:.1%}")

# ---------------------------------------------------------------------------
# Section 6: Per-Day & Duration Analysis
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 6: PER-DAY & TRADE DURATION ANALYSIS (training, best config)")
print("=" * 80)

if best_trades_train:
    df_bt = pd.DataFrame(best_trades_train)

    daily_counts = df_bt.groupby("date").size()
    print(f"\nTrades per session:")
    print(f"  min={daily_counts.min()}, p25={daily_counts.quantile(.25):.0f}, "
          f"median={daily_counts.median():.0f}, p75={daily_counts.quantile(.75):.0f}, "
          f"max={daily_counts.max()}")

    print("\nAvg bars_held by exit reason:")
    for reason, grp in df_bt.groupby("reason"):
        print(f"  {reason:15s}: avg={grp['bars_held'].mean():.1f}, "
              f"N={len(grp)}, WR={(grp['pnl']>0).mean():.1%}")

    print("\nExit reason breakdown:")
    rc = df_bt["reason"].value_counts()
    for r, c in rc.items():
        pct = c / len(df_bt)
        avg_pnl = df_bt[df_bt["reason"] == r]["pnl"].mean()
        print(f"  {r:15s}: {c:4d} ({pct:.1%}), avg_pnl=${avg_pnl:.0f}")

    daily_pnl = df_bt.groupby("date")["pnl"].sum()
    print("\nBest 5 sessions:")
    for d, v in daily_pnl.nlargest(5).items():
        print(f"  {d}: ${v:,.0f}")
    print("Worst 5 sessions:")
    for d, v in daily_pnl.nsmallest(5).items():
        print(f"  {d}: ${v:,.0f}")
else:
    print("No trades.")

# ---------------------------------------------------------------------------
# Section 7: Monte Carlo (Combine)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 7: COMBINE MONTE CARLO (10k paths x 60 days, training daily PnLs)")
print("=" * 80)

if best_trades_train:
    df_bt = pd.DataFrame(best_trades_train)
    daily_pnl_series = df_bt.groupby("date")["pnl"].sum().values

    np.random.seed(42)
    n_paths = 10_000
    n_days = 60
    trail_dd_limit = -2000.0
    daily_dd_limit = -1000.0
    profit_target = 3000.0

    passed = 0
    busted = 0
    finish_days = []

    for _ in range(n_paths):
        sample = np.random.choice(daily_pnl_series, size=n_days, replace=True)
        cum_pnl = 0.0
        peak_pnl = 0.0
        result = "timeout"
        day_result = 0
        for d_idx, dpnl in enumerate(sample):
            # daily loss limit
            if dpnl < daily_dd_limit:
                dpnl = daily_dd_limit
            cum_pnl += dpnl
            peak_pnl = max(peak_pnl, cum_pnl)
            trail_dd = cum_pnl - peak_pnl
            if trail_dd <= trail_dd_limit:
                result = "bust"
                day_result = d_idx + 1
                break
            if cum_pnl >= profit_target:
                result = "pass"
                day_result = d_idx + 1
                break
        else:
            day_result = n_days
        if result == "pass":
            passed += 1
            finish_days.append(day_result)
        elif result == "bust":
            busted += 1

    p_pass = passed / n_paths
    p_bust = busted / n_paths
    median_days = int(np.median(finish_days)) if finish_days else n_days

    print(f"  P(pass)    = {p_pass:.1%}")
    print(f"  P(bust)    = {p_bust:.1%}")
    print(f"  Median days to pass (of passers) = {median_days}")
else:
    p_pass = 0.0
    p_bust = 1.0
    median_days = 60
    print("  No trades — cannot run Monte Carlo.")

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

n_per_day = best_stats_train.get("N_per_day", 0)
wr = best_stats_train.get("WR", 0)
oos_wr_val = oos_stats.get("WR", 0)
wr_gap_val = abs(wr - oos_wr_val)

print(f"  N/day (train)   : {n_per_day:.2f}")
print(f"  WR (train)      : {wr:.1%}")
print(f"  Z mono          : {'PASS' if not (mono_fail_count >= 1 and not z_mono) else 'FAIL'}")
print(f"  OOS WR gap      : {wr_gap_val:.1%}")
print(f"  P(pass combine) : {p_pass:.1%}")

if (
    n_per_day >= 1.5
    and wr >= 0.52
    and mono_fail_count < 3
    and wr_gap_val <= 0.08
):
    verdict = "PASS"
elif (
    n_per_day >= 1.0
    and wr >= 0.48
    and mono_fail_count <= 2
):
    verdict = "MARGINAL"
else:
    verdict = "KILL"

print(f"\n  *** VERDICT: {verdict} ***")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

output = {
    "verdict": verdict,
    "best_config": {
        "phi": best_phi,
        "z_star": best_z_star,
        "sl_k": best_sl_k,
        "decay_bars": best_decay,
    },
    "train_stats": best_stats_train,
    "oos_stats": oos_stats,
    "mnq_stats": mnq_stats,
    "monotonicity": {
        "z_mono": bool(z_mono) if best_trades_train else False,
        "decay_mono": bool(d_mono) if best_trades_train else False,
        "p_mono": bool(p_mono) if best_trades_train else False,
        "fail_count": mono_fail_count,
    },
    "monte_carlo": {
        "p_pass": round(p_pass, 4),
        "p_bust": round(p_bust, 4),
        "median_days_to_pass": median_days,
    },
    "top15_sweep": sweep_df.head(15)
    .fillna(0)
    .to_dict(orient="records"),
}

out_path = ROOT / "rule_based_v1/diagnostics/lppf_results.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to: {out_path}")
