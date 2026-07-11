"""Parity check: confirm the live runner's decision logic reproduces v3's backtest.

Replays the OOS microstructure bars through the EXACT gates the live runner uses
(MLStrategyRunner._in_trading_window for the session, min_atr_pts, conf threshold,
and the pt/sl/lookahead exit logic), then compares the resulting trades to the
original validated v3 OOS run (54 trades, WR 53.7%).

A match confirms the deployed bundle + session bound + min-ATR filter faithfully
reproduce the validated edge. Run:
    python ml_intraday_v3/diagnostics/parity_check_v3_runner.py
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rule_based_v1"))
sys.path.insert(0, str(ROOT / "ml_intraday_v3"))

POINT_VALUE = 2.0
COMMISSION  = 0.62


def main():
    import importlib
    runner_mod = importlib.import_module("rule_based_v1.live.ml_strategy_runner")
    in_window = runner_mod.MLStrategyRunner._in_trading_window

    with open(ROOT / "rule_based_v1/models/ml_strategy_mnq_v3.pkl", "rb") as f:
        bundle = pickle.load(f)
    model       = bundle["model"]
    feat_cols   = bundle["feature_cols"]
    cfg         = bundle["config"]
    conf        = float(cfg["conf"])
    pt_mult     = float(cfg["pt"])
    sl_mult     = float(cfg["sl"])
    horizon     = int(cfg["lookahead"])
    min_atr     = float(cfg["min_atr_pts"])
    max_trades  = int(cfg["max_trades"])
    cooldown    = int(cfg.get("cooldown_bars", 3))

    # Match the original v3 OOS window for an apples-to-apples comparison
    orig_window = pd.read_parquet(ROOT / "ml_intraday_v3/results/ml_scalper_v3_oos_trades.parquet")
    orig_end = pd.to_datetime(orig_window["entry_time"]).max().normalize() + pd.Timedelta(days=1)

    from ml_intraday_v3.scripts.ml_scalper_v7 import build_features
    micro = pd.read_parquet(ROOT / "data/processed/mnq_microstructure_5min.parquet")
    micro.index = pd.to_datetime(micro.index, utc=True)
    feat = build_features(micro)
    oos = feat[(feat.index > pd.Timestamp(bundle["train_end"], tz="UTC"))
               & (feat.index < orig_end)].copy()
    print(f"Comparison window: {bundle['train_end']} -> {orig_end.date()} "
          f"(matched to original OOS)\n")

    valid = oos[feat_cols].notna().all(axis=1) & oos["atr"].notna()
    df = oos[valid].copy()
    X = df[feat_cols].astype(np.float32).values
    p_long = model.predict_proba(X)[:, 1]

    c = df["close"].values; h = df["high"].values; l = df["low"].values
    av = df["atr"].values; idx = df.index

    # Runner's risk overlay: 3-consecutive-loss circuit breaker halts the day,
    # 3-bar cooldown after each trade (matches RiskManager defaults).
    MAX_CONSEC_LOSS = 3
    trades = []
    daily_count: dict[str, int] = {}
    daily_consec_loss: dict[str, int] = {}
    in_trade = False; cooldown_left = 0
    ep = ea = 0.0; eb = 0
    for i in range(len(df)):
        ds = str(idx[i].date())
        daily_count.setdefault(ds, 0)
        daily_consec_loss.setdefault(ds, 0)
        if in_trade:
            ptp = ep + pt_mult * ea
            slp = ep - sl_mult * ea
            if h[i] >= ptp:        pts, why = pt_mult * ea, "PT"
            elif l[i] <= slp:      pts, why = -sl_mult * ea, "SL"
            elif (i - eb) >= horizon: pts, why = c[i] - ep, "TIME"
            else: continue
            net = pts * POINT_VALUE - 2 * COMMISSION
            trades.append({"entry_time": idx[eb], "exit_reason": why,
                           "pnl_pts": pts, "pnl_net": net})
            in_trade = False; cooldown_left = cooldown
            daily_consec_loss[ds] = daily_consec_loss[ds] + 1 if net < 0 else 0
            continue
        if cooldown_left > 0:
            cooldown_left -= 1
            continue
        if daily_consec_loss[ds] >= MAX_CONSEC_LOSS:  continue   # circuit breaker
        if daily_count[ds] >= max_trades:        continue
        if np.isnan(av[i]) or av[i] < min_atr:   continue
        if not in_window(idx[i]):                continue
        if p_long[i] >= conf:
            in_trade = True; ep, ea, eb = c[i], av[i], i
            daily_count[ds] += 1

    tdf = pd.DataFrame(trades)
    n = len(tdf)
    wr = (tdf["pnl_net"] > 0).mean() if n else 0
    pnl = tdf["pnl_net"].sum() if n else 0

    # Original validated v3 OOS (from saved trades, per-contract net)
    orig = pd.read_parquet(ROOT / "ml_intraday_v3/results/ml_scalper_v3_oos_trades.parquet")
    orig_n = len(orig)
    orig_pc_net = orig["pnl_pts"] * POINT_VALUE - 2 * COMMISSION
    orig_wr = (orig_pc_net > 0).mean()

    print("=== Runner-logic replay (bounded session + min_atr + conf) ===")
    print(f"  trades={n}  WR={wr:.3f}  per-contract net PnL=${pnl:,.2f}")
    print(f"  exit mix: {tdf['exit_reason'].value_counts().to_dict() if n else {}}")
    print()
    print("=== Original validated v3 OOS ===")
    print(f"  trades={orig_n}  WR={orig_wr:.3f}  per-contract net PnL=${orig_pc_net.sum():,.2f}")
    print()
    dn = abs(n - orig_n)
    print(f"PARITY: trade-count delta={dn} ({'PASS' if dn <= 3 else 'INVESTIGATE'}), "
          f"WR delta={abs(wr-orig_wr):.3f}")
    if dn <= 3 and abs(wr - orig_wr) < 0.05:
        print("=> Runner logic faithfully reproduces the validated v3 edge.")
    else:
        print("=> Divergence — investigate gating differences before live.")


if __name__ == "__main__":
    main()
