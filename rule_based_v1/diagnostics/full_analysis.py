"""
Full strategy analysis: ORB + VWAP MR across all available data.

Sections:
  1. 2026 YTD  — full deployed config (PrevVWAP + momentum + skip_gap + wide_OR)
  2. MNQ Aug-Dec 2025  — pre-2026 MNQ validation
  3. MES Oct 2024-Dec 2025  — alternate-period validation (S&P proxy)
  4. ES 2010-2025  — 15-year long-run test (raw ORB only; no PrevVWAP)
  5. Combined: ORB + VWAP MR (2026 YTD, deployed contracts)
  6. Monte Carlo  — simulate 13-week funded-account paths
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for p in [str(ROOT), str(RBV1)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
from utils.indicators import atr as calc_atr

# ── Constants (deployed config) ──────────────────────────────────────────────
POINT_VALUE   = 2.0
TICK_SIZE     = 0.25
COMMISSION    = 0.62
ORB_NC        = 10
VWAP_NC       = 5
OR_END        = "10:04"
MIN_OR_BARS   = 7
ATR_PERIOD    = 14
PT_ATR        = 3.0
SL_ATR        = 1.5
MAX_BARS      = 24
ENTRY_CUTOFF  = "12:00"
MAX_OR_ATR    = 5.0
SLIPPAGE_TCK  = 1   # 1 tick slippage on entry

VWAP_PT       = 2.0
VWAP_SL       = 1.5
VWAP_TBARS    = 12
VWAP_DIST     = 1.0
VWAP_MAX_DIST = 3.0
VWAP_MAX_MOVE = 2.0
VWAP_TSTART   = (10, 30)
VWAP_TEND     = (13, 30)

DATA = {
    "mnq_2026ytd": ROOT / "data/processed/mnq_2026ytd_databento_5min_rth.h5",
    "mnq_aug25":   ROOT / "data/processed/mnq_5min_aug25_mar26.h5",
    "mes_2024":    ROOT / "data/processed/MES_5min_Oct2024_Dec2025.parquet",
    "es_long":     ROOT / "data/processed/es_bars_2010_2025.h5",
}


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_bars(key: str) -> pd.DataFrame:
    p = DATA[key]
    if key == "mes_2024":
        df = pd.read_parquet(str(p))
    elif key == "es_long":
        df = pd.read_hdf(str(p), key="bars_5min")
        df = df.set_index(pd.to_datetime(df["timestamp"], unit="s", utc=True))
        df = df.drop(columns=["timestamp"])
    else:
        df = pd.read_hdf(str(p), key="bars_5min")
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")
    # RTH filter
    df = df[(df.index.hour >= 9) & (df.index.hour < 16)]
    return df.sort_index()


# ── Daily meta (PrevVWAP + momentum) ─────────────────────────────────────────
def build_daily_meta(bars: pd.DataFrame, lookback: int = 3) -> dict:
    """Returns {date: {prev_vwap_bull, momentum_ok, gap_pct}}"""
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    date_key = bars.index.map(lambda t: t.date())
    vwap = (tp * bars["volume"]).groupby(date_key).cumsum() / \
           bars["volume"].groupby(date_key).cumsum().replace(0, np.nan)
    daily_close = bars.groupby(date_key)["close"].last()
    daily_open  = bars.groupby(date_key)["open"].first()
    daily_vwap_eod = vwap.groupby(date_key).last()
    close_above_vwap = (daily_close > daily_vwap_eod)
    sorted_dates = sorted(set(bars.index.date))
    closes = daily_close.reindex(sorted_dates)

    meta = {}
    for i, d in enumerate(sorted_dates):
        prev = sorted_dates[i-1] if i > 0 else None
        pv_bull = bool(close_above_vwap.get(prev)) if prev else None
        # 3-day momentum
        if i >= lookback:
            ret3 = (closes.iloc[i-1] - closes.iloc[i-lookback-1]) / closes.iloc[i-lookback-1]
            mom_ok = (ret3 > -0.01)
        else:
            mom_ok = True
        # Gap pct (today open vs yesterday close)
        if prev and prev in daily_close.index and d in daily_open.index:
            gap = (float(daily_open[d]) - float(daily_close[prev])) / float(daily_close[prev])
        else:
            gap = 0.0
        meta[d] = {"prev_vwap_bull": pv_bull, "momentum_ok": mom_ok, "gap_pct": gap}
    return meta


# ── Core ORB simulator (per bar, no equity tracking) ─────────────────────────
def sim_orb(
    bars: pd.DataFrame,
    nc: int = ORB_NC,
    use_filters: bool = True,
    meta: dict | None = None,
    skip_gap_up: float = 0.005,
    max_or_range_atr: float = MAX_OR_ATR,
    label: str = "",
) -> list[dict]:
    atr_s = calc_atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    comm_rt = 2 * COMMISSION * nc
    slip = SLIPPAGE_TCK * TICK_SIZE

    trades = []
    dates = sorted(set(bars.index.date))

    for d in dates:
        dm = bars[np.array(bars.index.date) == d]
        da = atr_s[np.array(atr_s.index.date) == d]
        if len(dm) < 15:
            continue

        # Filters
        if use_filters and meta:
            m = meta.get(d, {})
            if m.get("prev_vwap_bull") is False:
                continue   # bearish PrevVWAP (no override here for simplicity)
            if not m.get("momentum_ok", True):
                continue
            gap = m.get("gap_pct", 0.0)
            if gap > skip_gap_up:
                continue

        # OR window
        or_mask = (dm.index.hour == 9) | ((dm.index.hour == 10) & (dm.index.minute <= 4))
        or_bars = dm[or_mask]
        if len(or_bars) < MIN_OR_BARS:
            continue

        or_high = or_bars["high"].max()
        or_low  = or_bars["low"].min()
        or_range = or_high - or_low
        atr_val  = float(da.iloc[min(len(or_bars)-1, len(da)-1)])
        if atr_val <= 0 or np.isnan(atr_val):
            continue

        if or_range < 0.3 * atr_val:
            continue
        if or_range > max_or_range_atr * atr_val:
            continue

        # Entry window
        entry_mask = ((dm.index.hour > 10) | ((dm.index.hour == 10) & (dm.index.minute > 4))) \
                     & (dm.index.hour < 12)
        ew = dm[entry_mask]
        if len(ew) == 0:
            continue

        prev_close = float(or_bars["close"].iloc[-1])
        for j, (ts, row) in enumerate(ew.iterrows()):
            if row["close"] > or_high and prev_close <= or_high:
                ep = or_high + TICK_SIZE + slip
                pt = ep + PT_ATR * atr_val
                sl = ep - SL_ATR * atr_val
                pnl, reason = None, None
                remain = ew.iloc[j+1:]
                for k, (ts2, r2) in enumerate(remain.iterrows()):
                    if k >= MAX_BARS:
                        pnl = (r2["close"] - ep) * nc * POINT_VALUE - comm_rt
                        reason = "time_stop"
                        break
                    if r2["low"] <= sl:
                        pnl = (sl - TICK_SIZE - ep) * nc * POINT_VALUE - comm_rt
                        reason = "stop_loss"
                        break
                    if r2["high"] >= pt:
                        pnl = (pt - TICK_SIZE - ep) * nc * POINT_VALUE - comm_rt
                        reason = "profit_target"
                        break
                else:
                    if len(remain) > 0:
                        pnl = (remain.iloc[-1]["close"] - ep) * nc * POINT_VALUE - comm_rt
                        reason = "session_close"
                    else:
                        pnl, reason = 0.0, "session_close"
                trades.append({"date": d, "pnl": pnl, "won": pnl > 0,
                                "reason": reason, "or_range_atr": or_range/atr_val,
                                "atr": atr_val, "entry": ep})
                break
            prev_close = float(row["close"])

    return trades


# ── VWAP MR simulator ─────────────────────────────────────────────────────────
def sim_vwap(bars: pd.DataFrame, nc: int = VWAP_NC) -> list[dict]:
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    date_key = bars.index.map(lambda t: t.date())
    vwap_s = (tp * bars["volume"]).groupby(date_key).cumsum() / \
              bars["volume"].groupby(date_key).cumsum().replace(0, np.nan)
    atr_s = calc_atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    day_open_s = bars.groupby(date_key)["open"].transform("first")

    comm_rt = 2 * COMMISSION * nc
    trades = []
    dates = sorted(set(bars.index.date))

    for d in dates:
        dm   = bars[np.array(bars.index.date) == d]
        dv   = vwap_s[np.array(vwap_s.index.date) == d]
        da   = atr_s[np.array(atr_s.index.date) == d]
        ddop = day_open_s[np.array(day_open_s.index.date) == d]
        day_trades = 0

        pos = None
        for i in range(1, len(dm)):
            ts  = dm.index[i]
            m   = ts.hour * 60 + ts.minute
            if not (VWAP_TSTART[0]*60+VWAP_TSTART[1] <= m <= VWAP_TEND[0]*60+VWAP_TEND[1]):
                if pos is not None:
                    # session close
                    ep = float(dm["close"].iloc[i])
                    pnl = (ep - pos["entry"]) * pos["dir"] * nc * POINT_VALUE - comm_rt
                    trades.append({"date": d, "pnl": pnl, "won": pnl>0, "reason": "session_close"})
                    pos = None
                continue

            c   = float(dm["close"].iloc[i])
            pc  = float(dm["close"].iloc[i-1])
            a   = float(da.iloc[i])
            v   = float(dv.iloc[i])
            dop = float(ddop.iloc[i])
            is_last = (i == len(dm)-1)

            if np.isnan(a) or a <= 0 or np.isnan(v):
                continue

            dev = (c - v) / a
            mov = (c - dop) / a

            # Manage open position
            if pos is not None:
                h = float(dm["high"].iloc[i])
                lo = float(dm["low"].iloc[i])
                exited = False
                ep_exit = 0.0
                if is_last or i >= pos["stop_i"]:
                    ep_exit = c
                    exited = True
                elif pos["dir"] == 1:
                    if lo <= pos["sl"]: ep_exit = pos["sl"] - TICK_SIZE; exited = True
                    elif h >= pos["pt"]: ep_exit = pos["pt"] - TICK_SIZE; exited = True
                elif pos["dir"] == -1:
                    if h >= pos["sl"]: ep_exit = pos["sl"] + TICK_SIZE; exited = True
                    elif lo <= pos["pt"]: ep_exit = pos["pt"] + TICK_SIZE; exited = True
                if exited:
                    pnl = (ep_exit - pos["entry"]) * pos["dir"] * nc * POINT_VALUE - comm_rt
                    trades.append({"date": d, "pnl": pnl, "won": pnl>0, "reason": "exit"})
                    pos = None

            if pos is not None or day_trades >= 2:
                continue

            if abs(dev) > VWAP_MAX_DIST:
                continue

            # LONG: below VWAP, last bar up, not in strong downtrend
            if dev <= -VWAP_DIST and c > pc and mov > -VWAP_MAX_MOVE:
                ep = c + TICK_SIZE
                pos = {"entry": ep, "dir": 1,
                       "pt": ep + VWAP_PT*a, "sl": ep - VWAP_SL*a,
                       "stop_i": i + VWAP_TBARS}
                day_trades += 1

    return trades


# ── Stats helper ─────────────────────────────────────────────────────────────
def stats(trades: list[dict], label: str, n_days: int) -> dict:
    if not trades:
        print(f"  {label}: NO TRADES")
        return {}
    arr = np.array([t["pnl"] for t in trades])
    wins = arr[arr > 0]
    loss = arr[arr <= 0]
    equity = np.concatenate([[0], np.cumsum(arr)])
    dd = float((equity - np.maximum.accumulate(equity)).min())
    n_weeks = n_days / 5
    wk = arr.sum() / n_weeks if n_weeks > 0 else 0
    pf = wins.sum() / abs(loss.sum()) if len(loss) > 0 else 99.0
    sharpe = float(arr.mean() / arr.std() * np.sqrt(252/n_days*len(arr))) if arr.std() > 0 else 0
    wr = (arr > 0).mean()
    return {
        "label": label, "n": len(arr), "wr": wr, "pnl": arr.sum(),
        "weekly": wk, "avg_win": wins.mean() if len(wins) else 0,
        "avg_loss": loss.mean() if len(loss) else 0,
        "max_dd": dd, "pf": pf, "sharpe": sharpe,
        "n_weeks": n_weeks, "trades_arr": arr,
    }


def print_stats(s: dict):
    if not s:
        return
    print(f"\n  {'─'*55}")
    print(f"  {s['label']}")
    print(f"  {'─'*55}")
    print(f"  Trades     : {s['n']}  ({s['n']/s['n_weeks']:.1f}/wk)")
    print(f"  Win Rate   : {s['wr']:.1%}")
    print(f"  Total PnL  : ${s['pnl']:,.0f}")
    print(f"  Avg Win    : ${s['avg_win']:,.0f}  |  Avg Loss: ${s['avg_loss']:,.0f}")
    print(f"  $/week     : ${s['weekly']:,.0f}")
    print(f"  Max DD     : ${s['max_dd']:,.0f}")
    print(f"  Profit Fac : {s['pf']:.2f}")
    print(f"  Sharpe     : {s['sharpe']:.2f}")


# ── Monte Carlo ───────────────────────────────────────────────────────────────
def monte_carlo(trades_arr: np.ndarray, n_weeks: int = 13,
                trades_per_week: float = 1.5, n_paths: int = 20_000,
                profit_target: float = 3_000, max_dd_limit: float = 2_000,
                label: str = "") -> dict:
    rng = np.random.default_rng(42)
    n_trades = int(round(n_weeks * trades_per_week))
    paths = rng.choice(trades_arr, size=(n_paths, n_trades), replace=True)
    cum   = np.cumsum(paths, axis=1)
    peak  = np.maximum.accumulate(cum, axis=1)
    dd    = (cum - peak).min(axis=1)
    final = cum[:, -1]

    p_pass  = float((final >= profit_target).mean())
    p_blown = float((dd <= -max_dd_limit).mean())
    p_both  = float(((final >= profit_target) & (dd > -max_dd_limit)).mean())
    med_pnl = float(np.median(final))
    p5_pnl  = float(np.percentile(final, 5))
    p95_pnl = float(np.percentile(final, 95))
    med_dd  = float(np.median(dd))
    p95_dd  = float(np.percentile(dd, 5))   # worst 5%

    print(f"\n  {'═'*55}")
    print(f"  MONTE CARLO — {label}")
    print(f"  {n_paths:,} paths × {n_weeks}wk × {trades_per_week:.1f} trades/wk = {n_trades} trades")
    print(f"  {'═'*55}")
    print(f"  P(reach ${profit_target:,.0f} profit target) : {p_pass:.1%}")
    print(f"  P(breach ${max_dd_limit:,.0f} max DD)        : {p_blown:.1%}")
    print(f"  P(pass AND survive DD)              : {p_both:.1%}")
    print(f"  Median 13-wk PnL : ${med_pnl:,.0f}  [p5=${p5_pnl:,.0f}  p95=${p95_pnl:,.0f}]")
    print(f"  Median max DD    : ${med_dd:,.0f}  [worst 5%=${p95_dd:,.0f}]")
    return {"p_pass": p_pass, "p_blown": p_blown, "med_pnl": med_pnl, "med_dd": med_dd}


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():

    # ── 1. 2026 YTD with full deployed filters ─────────────────────────────
    print(f"\n{'═'*60}")
    print("1. MNQ 2026 YTD  (Jan–May 2026)  — DEPLOYED CONFIG")
    print(f"{'═'*60}")
    bars26 = load_bars("mnq_2026ytd")
    meta26 = build_daily_meta(bars26)
    n_days26 = len(set(bars26.index.date))

    orb26 = sim_orb(bars26, nc=ORB_NC, use_filters=True, meta=meta26, label="ORB 2026 filtered")
    s_orb26 = stats(orb26, f"ORB 2026 YTD  (10 MNQ, full filters, +slippage)", n_days26)
    print_stats(s_orb26)

    orb26_raw = sim_orb(bars26, nc=ORB_NC, use_filters=False, label="ORB 2026 raw")
    s_orb26r = stats(orb26_raw, f"ORB 2026 YTD  (10 MNQ, NO filters)", n_days26)
    print_stats(s_orb26r)

    vwap26 = sim_vwap(bars26, nc=VWAP_NC)
    s_v26 = stats(vwap26, f"VWAP MR 2026 YTD  (5 MNQ)", n_days26)
    print_stats(s_v26)

    # Combined
    all26 = orb26 + vwap26
    all26.sort(key=lambda t: t["date"])
    s_comb26 = stats(all26, f"COMBINED ORB+VWAP 2026 YTD  (10+5 MNQ)", n_days26)
    print_stats(s_comb26)

    # ── 2. MNQ Aug-Dec 2025 ────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("2. MNQ Aug–Mar 2025-26  — PRE-2026 VALIDATION")
    print(f"{'═'*60}")
    bars25 = load_bars("mnq_aug25")
    bars25_pre26 = bars25[bars25.index < pd.Timestamp("2026-01-01", tz="US/Eastern")]
    meta25 = build_daily_meta(bars25_pre26)
    n_days25 = len(set(bars25_pre26.index.date))

    orb25 = sim_orb(bars25_pre26, nc=ORB_NC, use_filters=True, meta=meta25, label="ORB 2025")
    s_orb25 = stats(orb25, f"ORB Aug–Dec 2025  (10 MNQ, full filters)", n_days25)
    print_stats(s_orb25)

    vwap25 = sim_vwap(bars25_pre26, nc=VWAP_NC)
    s_v25 = stats(vwap25, f"VWAP MR Aug–Dec 2025  (5 MNQ)", n_days25)
    print_stats(s_v25)

    comb25 = orb25 + vwap25
    comb25.sort(key=lambda t: t["date"])
    s_comb25 = stats(comb25, f"COMBINED ORB+VWAP Aug–Dec 2025", n_days25)
    print_stats(s_comb25)

    # ── 3. MES Oct 2024-Dec 2025 ───────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("3. MES Oct 2024–Dec 2025  — S&P PROXY / OUT-OF-SAMPLE INSTRUMENT")
    print(f"{'═'*60}")
    bars_mes = load_bars("mes_2024")
    meta_mes = build_daily_meta(bars_mes)
    n_days_mes = len(set(bars_mes.index.date))

    orb_mes = sim_orb(bars_mes, nc=ORB_NC, use_filters=True, meta=meta_mes, label="ORB MES")
    s_mes = stats(orb_mes, f"ORB MES Oct'24–Dec'25  (10c, full filters)", n_days_mes)
    print_stats(s_mes)

    # ── 4. ES 2010-2025 long-run ───────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("4. ES 2010–2025  — 15-YEAR LONG-RUN (no PrevVWAP; raw ORB only)")
    print(f"{'═'*60}")
    try:
        bars_es = load_bars("es_long")
        # Scale to ES: point_value=50, tick=0.25, approx equiv = 1 ES = 10 MNQ roughly
        # Use nc=1 ES and show per-contract stats
        orb_es = sim_orb(bars_es, nc=1, use_filters=False, label="ORB ES raw")
        s_es = stats(orb_es, "ORB ES 2010–2025  (1c, raw, no filters)", len(set(bars_es.index.date)))
        # Override point_value in stats display (ES=50 not MNQ=2)
        # The pnl above was computed with POINT_VALUE=2; scale for display by 50/2=25
        scale = 50.0 / 2.0
        s_es_disp = s_es.copy()
        for k in ["pnl","weekly","avg_win","avg_loss","max_dd"]:
            s_es_disp[k] = s_es[k] * scale
        s_es_disp["trades_arr"] = s_es["trades_arr"] * scale
        s_es_disp["label"] = "ORB ES 2010–2025  (1 ES contract, point_val=50)"
        print_stats(s_es_disp)

        # Year-by-year
        print(f"\n  Year-by-year ORB WR (ES, 1c, raw):")
        es_df = pd.DataFrame(orb_es)
        if not es_df.empty:
            es_df["year"] = pd.to_datetime(es_df["date"]).dt.year
            for yr, grp in es_df.groupby("year"):
                wr = grp["won"].mean()
                pnl = grp["pnl"].sum() * scale
                bar = "#" * int(wr * 30)
                print(f"  {yr}  WR={wr:.1%}  PnL=${pnl:>8,.0f}  n={len(grp):>3}  {bar}")
    except Exception as e:
        print(f"  ES data load failed: {e}")

    # ── 5. Monte Carlo ─────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("5. MONTE CARLO SIMULATIONS")
    print(f"{'═'*60}")

    if s_orb26.get("trades_arr") is not None:
        arr = s_orb26["trades_arr"]
        tpw = s_orb26["n"] / s_orb26["n_weeks"]
        monte_carlo(arr, n_weeks=13, trades_per_week=tpw, label="ORB 2026 (filtered, 10 MNQ)",
                    profit_target=3000, max_dd_limit=2000)

    if s_comb26.get("trades_arr") is not None:
        arr_c = s_comb26["trades_arr"]
        tpw_c = s_comb26["n"] / s_comb26["n_weeks"]
        monte_carlo(arr_c, n_weeks=13, trades_per_week=tpw_c, label="COMBINED ORB+VWAP 2026",
                    profit_target=3000, max_dd_limit=2000)

    # All available MNQ trades combined (2025 + 2026) for larger sample
    all_mnq_orb = orb25 + orb26
    if len(all_mnq_orb) >= 10:
        arr_all = np.array([t["pnl"] for t in all_mnq_orb])
        tpw_all = len(all_mnq_orb) / ((n_days25 + n_days26) / 5)
        monte_carlo(arr_all, n_weeks=13, trades_per_week=tpw_all,
                    label="ORB MNQ Aug'25–May'26  (largest sample, 10 MNQ)",
                    profit_target=3000, max_dd_limit=2000)

    # ── 6. Summary ─────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("SUMMARY TABLE")
    print(f"{'═'*60}")
    print(f"  {'Period':<30} {'N':>4} {'WR':>7} {'$/wk':>8} {'MaxDD':>9} {'PF':>6} {'Sharpe':>7}")
    print(f"  {'-'*72}")
    for s in [s_orb26, s_comb26, s_orb25, s_comb25, s_mes]:
        if s:
            print(f"  {s['label'][:30]:<30} {s['n']:>4} {s['wr']:>7.1%} "
                  f"${s['weekly']:>7,.0f} ${s['max_dd']:>8,.0f} "
                  f"{s['pf']:>6.2f} {s['sharpe']:>7.2f}")


if __name__ == "__main__":
    main()
