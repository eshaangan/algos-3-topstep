"""Micro Nikkei (USD) — Asia-session (Tokyo clock) ORB research.

Uses OR window in Japan local time (default 09:00–10:04 JST), ETH 5m bars from Databento
(no US RTH filter). Backtest calendar + session flatten use Tokyo dates / 15:55 JST cut.

Run from repo root:
  python3 rule_based_v1/diagnostics/orb_mnk_asia_research.py
  python3 rule_based_v1/diagnostics/orb_mnk_asia_research.py --backtest-only

Live note: TopstepX bar fetch may be US RTH-only; Asia ORB needs extended-hours bars.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for p in (str(ROOT), str(RBV1)):
    if p not in sys.path:
        sys.path.insert(0, p)

from diagnostics.ifr_backtest import (  # noqa: E402
    _filter_ytd,
    _load_bars,
    _topstep_check,
    fetch_and_save_eth,
    run_backtest_instrument,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MNK_ETH_PATH = ROOT / "data" / "processed" / "mnk_eth_5min.h5"
RESULTS_JSON = ROOT / "rule_based_v1" / "diagnostics" / "orb_mnk_asia_results.json"

MNK_SPEC = {
    "symbol": "MNK.c.0",
    "point_value": 0.5,
    "tick_size": 5.0,
    "tick_value": 2.5,
    "commission": 0.62,
    "slippage_ticks": 1,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="MNK Tokyo-session ORB on ETH 5m bars")
    parser.add_argument("--backtest-only", action="store_true")
    parser.add_argument("--start", default="2024-06-01")
    parser.add_argument("--end", default="2026-04-15T23:59:59+00:00")
    args = parser.parse_args()

    df = None
    if not args.backtest_only:
        try:
            df = fetch_and_save_eth("MNK.c.0", MNK_ETH_PATH, args.start, args.end)
        except Exception as e:
            logger.error("ETH fetch failed: %s", e)

    if df is None and MNK_ETH_PATH.exists():
        logger.info("Loading cached ETH MNK …")
        df = _load_bars(MNK_ETH_PATH, "bars_5min_eth")

    if df is None or len(df) == 0:
        logger.error("No ETH MNK data. Set DATABENTO_API_KEY or fetch first.")
        sys.exit(1)

    df = _filter_ytd(df, args.start)
    logger.info("MNK ETH 5m: %s bars  %s → %s", len(df), df.index[0], df.index[-1])

    # Tokyo cash-style OR (09:00–10:04 JST), entries through noon JST; flatten sim @ 15:55 JST
    r = run_backtest_instrument(
        df,
        MNK_SPEC,
        1,
        session_timezone="Asia/Tokyo",
        calendar_mode="tokyo",
        tokyo_flat_time="15:55",
        session_start_time="09:00",
        or_end_time="10:04",
        min_or_bars=7,
        min_range_atr=0.30,
        entry_cutoff="12:00",
        pt_mult=3.5,
        sl_mult=1.75,
        long_only=True,
    )
    tc = _topstep_check(r)

    print("\n" + "=" * 72)
    print("  MNK — Asia (Tokyo) ORB  |  OR 09:00–10:04 JST  |  PT=3.5  SL=1.75 ATR")
    print("=" * 72)
    print(f"  Trades: {r['num_trades']}  PnL=${r['total_pnl']:,.0f}  Sharpe={r['sharpe']:.2f}")
    print(f"  MaxDD=${r['max_drawdown']:,.0f}  combine_heuristic={tc['combine_pass_likely']}")
    print("=" * 72 + "\n")

    out = {
        "mode": "asia_tokyo_orb",
        "or_window_jst": "09:00-10:04",
        "entry_cutoff_jst": "12:00",
        "flatten_cut_jst": "15:55",
        "period_start": args.start,
        "period_end": str(args.end),
        "results": {k: r[k] for k in r if k != "trades"},
        "topstep_check": tc,
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info("Wrote %s", RESULTS_JSON)


if __name__ == "__main__":
    main()
