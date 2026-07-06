"""Paper-trade imb_fade_midday live — forward out-of-sample validation, no money.

Reads the parquet chunks the L2 recorder is already writing (NO second Rithmic
connection), rebuilds 1-min bars, computes the frozen signal, and logs hypothetical
trades at real quoted prices. Self-contained on purpose: only pandas/pyarrow, no
repo imports, so it can run on the remote Mac next to the recorder.

FROZEN signal (must match research_l2.py sig_imb_fade_midday @ z=1.5, 1min):
  - l1_imb z-score, rolling 60 bars (min_periods 30)
  - z >= +1.5  -> SHORT (fade bid-heavy book);  z <= -1.5 -> LONG
  - active bars: 12:00-14:59 ET only
FROZEN sim params (match the gate's SimParams):
  entry next bar open, PT=1.5*ATR, SL=1.0*ATR (both-hit -> stop, conservative),
  horizon 10 bars, cooldown 1 bar, max 8 trades/day, 2 MNQ ($2/pt),
  commission $0.62/side/contract, 1 tick slippage per side on the mid-based fills.
Also logs the ACTUAL quoted bid/ask + spread at signal time -> real cost read.

    python paper_fade_runner.py --raw-dir <l2_raw> --out-dir <logs>
State in <out-dir>/paper_fade_state.json; trades in paper_fade_trades.jsonl.
Restart-safe: bars are rebuilt from today's chunks; processed bar keys are kept in
state so a restart never double-enters.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ── frozen parameters (DO NOT TUNE — pre-registered) ─────────────────────────
Z_ENTRY = 1.5
Z_WIN, Z_MINP = 60, 30
ET = "America/New_York"
ACTIVE_HOURS = (12, 14)            # 12:00-14:59 ET signal window
PT_ATR, SL_ATR = 1.5, 1.0
HORIZON_BARS, COOLDOWN_BARS = 10, 1
MAX_TRADES_DAY = 8
N_CONTRACTS, POINT_VALUE, TICK = 2, 2.0, 0.25
COMMISSION_SIDE = 0.62             # per side per contract
SLIP_TICKS = 1                     # per side, on top of mid


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}Z] {msg}", flush=True)


# ── data: recorder chunks -> two-sided BBO -> 1-min bars ─────────────────────

def load_today_bbo(raw_dir: str, day: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(raw_dir, f"bbo_MNQ_{day}_*.parquet")))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception:
            continue  # torn chunk mid-write; next cycle gets it
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).sort_values("recv_ns")
    # one-sided updates -> ffill each side, drop rows before both sides exist
    for side in ("bid", "ask"):
        absent = df[f"{side}_px"].isna()
        for col in (f"{side}_px", f"{side}_sz"):
            df.loc[absent, col] = np.nan
            df[col] = df[col].ffill()
    df = df.dropna(subset=["bid_px", "ask_px"])
    df["ts"] = pd.to_datetime(df["recv_ns"], unit="ns", utc=True)
    return df


def build_bars(bbo: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame({"ts": bbo["ts"]})
    s["mid"] = (bbo["bid_px"] + bbo["ask_px"]) / 2
    s["spread"] = bbo["ask_px"] - bbo["bid_px"]
    den = (bbo["bid_sz"] + bbo["ask_sz"]).replace(0, np.nan)
    s["l1_imb"] = (bbo["bid_sz"] - bbo["ask_sz"]) / den
    g = s.set_index("ts").resample("1min")
    bars = pd.DataFrame({
        "open": g["mid"].first(), "high": g["mid"].max(),
        "low": g["mid"].min(), "close": g["mid"].last(),
        "l1_imb": g["l1_imb"].mean(), "spread": g["spread"].mean(),
        "n_snap": g["mid"].count(),
    }).dropna(subset=["close"])
    h, l, c = bars["high"], bars["low"], bars["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    bars["atr"] = tr.ewm(span=14, min_periods=14).mean()
    zi = bars["l1_imb"]
    bars["z"] = (zi - zi.rolling(Z_WIN, min_periods=Z_MINP).mean()) / \
                zi.rolling(Z_WIN, min_periods=Z_MINP).std()
    return bars


def signal_at(bar) -> int:
    """Frozen fade rule on a CLOSED bar; 0 outside the midday ET window."""
    ts_et = bar.name.tz_convert(ET)
    if not (ACTIVE_HOURS[0] <= ts_et.hour <= ACTIVE_HOURS[1]):
        return 0
    z = bar["z"]
    if not np.isfinite(z) or not np.isfinite(bar["atr"]):
        return 0
    if z >= Z_ENTRY:
        return -1          # bid-heavy book -> fade -> SHORT
    if z <= -Z_ENTRY:
        return 1
    return 0


# ── state ─────────────────────────────────────────────────────────────────────

def load_state(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"open_trade": None, "processed": [], "day": None, "trades_today": 0,
            "cooldown_until": None}


def save_state(path: str, st: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1, default=str)
    os.replace(tmp, path)


def append_trade(path: str, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


# ── trade lifecycle on closed bars ────────────────────────────────────────────

def costs() -> float:
    return 2 * COMMISSION_SIDE * N_CONTRACTS + 2 * SLIP_TICKS * TICK * POINT_VALUE * N_CONTRACTS


def manage_open(trade: dict, bars: pd.DataFrame) -> dict | None:
    """Walk closed bars after entry; return closed-trade record or None if open."""
    after = bars[bars.index > pd.Timestamp(trade["entry_bar"])]
    d, e = trade["dir"], trade["entry_px"]
    pt = e + d * PT_ATR * trade["atr"]
    sl = e - d * SL_ATR * trade["atr"]
    for i, (ts, b) in enumerate(after.iterrows(), start=1):
        hit_pt = b["high"] >= pt if d > 0 else b["low"] <= pt
        hit_sl = b["low"] <= sl if d > 0 else b["high"] >= sl
        if hit_sl:                                   # both-hit -> stop (conservative)
            return _close(trade, ts, sl, "stop")
        if hit_pt:
            return _close(trade, ts, pt, "target")
        if i >= HORIZON_BARS:
            return _close(trade, ts, b["close"], "time")
    return None


def _close(trade: dict, ts, px: float, reason: str) -> dict:
    gross = trade["dir"] * (px - trade["entry_px"]) * POINT_VALUE * N_CONTRACTS
    trade.update(exit_bar=str(ts), exit_px=round(px, 2), exit_reason=reason,
                 pnl=round(gross - costs(), 2))
    return trade


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--once", action="store_true", help="single cycle (testing)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    state_p = os.path.join(args.out_dir, "paper_fade_state.json")
    trades_p = os.path.join(args.out_dir, "paper_fade_trades.jsonl")
    st = load_state(state_p)
    log(f"paper fade runner up — raw={args.raw_dir} out={args.out_dir}")

    while True:
        now_et = datetime.now(timezone.utc).astimezone().astimezone()
        now_et = pd.Timestamp.now(tz=ET)
        day = f"{now_et:%Y%m%d}"
        if st["day"] != day:                          # new session reset
            st.update(day=day, trades_today=0, processed=[], cooldown_until=None)

        # sleep hard outside 11:00-15:45 ET (need ~60 bars warmup before 12:00)
        if not (11 <= now_et.hour < 16):
            if st["open_trade"] is None and not args.once:
                save_state(state_p, st)
                time.sleep(600)
                continue

        try:
            bbo = load_today_bbo(args.raw_dir, day)
            bars = build_bars(bbo) if not bbo.empty else pd.DataFrame()
        except Exception as e:
            log(f"data error: {e}")
            bars = pd.DataFrame()

        if len(bars) >= 2:
            closed = bars.iloc[:-1]                   # last bar may be forming

            # 1) manage open paper trade
            if st["open_trade"]:
                done = manage_open(st["open_trade"], closed)
                if done:
                    append_trade(trades_p, done)
                    log(f"CLOSED {done['exit_reason']} pnl=${done['pnl']:+.2f} "
                        f"(dir={done['dir']} entry={done['entry_px']})")
                    st["open_trade"] = None
                    st["cooldown_until"] = str(closed.index[-1] +
                                               pd.Timedelta(minutes=COOLDOWN_BARS))

            # 2) look for entry on the latest closed bar
            sig_bar = closed.iloc[-1]
            key = str(sig_bar.name)
            cooldown_ok = (st["cooldown_until"] is None or
                           sig_bar.name >= pd.Timestamp(st["cooldown_until"]))
            if (st["open_trade"] is None and key not in st["processed"]
                    and st["trades_today"] < MAX_TRADES_DAY and cooldown_ok):
                d = signal_at(sig_bar)
                if d != 0:
                    # entry = next bar open ~ current mid; log real quotes too
                    last_q = bbo.iloc[-1]
                    entry = float(sig_bar["close"])
                    st["open_trade"] = {
                        "entry_bar": key, "dir": int(d),
                        "entry_px": round(entry, 2),
                        "atr": float(sig_bar["atr"]), "z": round(float(sig_bar["z"]), 2),
                        "quote_bid": float(last_q["bid_px"]), "quote_ask": float(last_q["ask_px"]),
                        "quote_spread": round(float(last_q["ask_px"] - last_q["bid_px"]), 2),
                        "bar_avg_spread": round(float(sig_bar["spread"]), 3),
                    }
                    st["trades_today"] += 1
                    log(f"ENTER {'LONG' if d > 0 else 'SHORT'} @~{entry:.2f} z={sig_bar['z']:+.2f} "
                        f"quoted {last_q['bid_px']}/{last_q['ask_px']} "
                        f"(trade {st['trades_today']}/{MAX_TRADES_DAY})")
                    append_trade(trades_p, {**st["open_trade"], "event": "entry"})
            st["processed"] = (st["processed"] + [key])[-500:]

        save_state(state_p, st)
        if args.once:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
