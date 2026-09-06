"""Timestamp-convention guard for MNQ bar files, and the account-rules loader.

TWO MNQ CONVENTIONS EXIST IN THIS REPO and mixing them shifts every winter
entry and exit by one hour, silently:

  data/processed/mnq_1m_all.parquet   true DST-aware America/Chicago
  data/hist_1m24v/m_*.parquet         raw Rithmic, fixed UTC-5 (no DST)

`replay_hold_strategies.load_bars()` -- the function the LIVE runners are
reconciled against -- hardcodes the fixed -05:00 offset AND accepts a
consolidated ".parquet" path, so pointing it at the already-converted file
applies the correction twice. An external audit traced bad statistics to exactly
this class of error on 2026-09-05.

`assert_et_index` is the cheap check: RTH opens at 09:30 ET all year, so the
opening-minute volume spike must land at 09:3x ET in BOTH halves of the year. A
fixed-offset file mislabelled as local time puts the winter spike at 09:0x.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

RULES = Path(__file__).resolve().parents[1] / "configs" / "account_rules.yaml"
FIRMS = Path(__file__).resolve().parents[1] / "configs" / "prop_firm_rules.yaml"


def load_firm_rules(firm: str, path: Path | None = None) -> dict:
    """One surveyed prop firm's rule set, in the shape check_window_legal() expects.

    Track B exists because the 16:45 flatten -- not the statistics -- is what kills
    this project's validated book. Comparing firms therefore has to be done with the
    same machine-checked legality test, not by reading marketing pages.
    """
    cfg = yaml.safe_load((path or FIRMS).read_text())
    f = dict(cfg["firms"][firm])
    f["firm"] = firm
    if not f.get("verified", False):
        print(f"  [{firm}: web-sourced, NOT confirmed against an account dashboard -- "
              "P(pass) is provisional]")
    return f


def load_account_rules(path: Path | None = None, account: str | None = None) -> dict:
    """Account rules from the single source of truth. Never hardcode these."""
    cfg = yaml.safe_load((path or RULES).read_text())
    name = account or cfg["active_account"]
    acct = dict(cfg["accounts"][name])
    acct.update({"account": name,
                 "consistency": cfg["consistency"]["max_best_day_fraction"],
                 "flatten_et": cfg["session"]["mandatory_flatten_et"],
                 "resumes_et": cfg["session"]["trading_resumes_et"],
                 "overnight_permitted": cfg["session"]["overnight_holding_permitted"],
                 "verified": cfg["verified_against_account_dashboard"]})
    if not acct["verified"]:
        print("  [account_rules.yaml is web-sourced and NOT yet confirmed against the "
              "account dashboard -- treat every P(pass) as provisional]")
    return acct


def assert_et_index(index: pd.DatetimeIndex, volume: pd.Series | None = None,
                    label: str = "bars") -> None:
    """Raise if the index does not look like true ET with DST handled correctly."""
    if index.tz is None:
        raise AssertionError(f"{label}: index is tz-naive; localize before use")
    et = index.tz_convert("America/New_York")
    if volume is None:
        return
    v = pd.Series(volume.to_numpy(), index=et)
    bad = []
    for season, months in (("winter", [12, 1, 2]), ("summer", [6, 7, 8])):
        s = v[v.index.month.isin(months)]
        if s.empty:
            continue
        hour9 = s[s.index.hour == 9]
        if hour9.empty:
            continue
        peak = int(hour9.groupby(hour9.index.minute).sum().idxmax())
        if not 29 <= peak <= 33:
            bad.append(f"{season} 09:{peak:02d}")
    if bad:
        raise AssertionError(
            f"{label}: RTH-open volume spike at {', '.join(bad)} ET, expected 09:30-09:33 "
            "in BOTH seasons. The timestamp convention is almost certainly wrong -- see "
            "the two-convention note at the top of tzguard.py.")


def check_window_legal(entry_et: str, exit_et: str, spans_days: int,
                       rules: dict) -> tuple[bool, str]:
    """Does a hold cross the mandatory flatten, and can it actually be entered?

    TWO conditions, not one. An earlier version checked only the flatten crossing and
    therefore reported a 17:00 ET entry as LEGAL. It is not: the platform flattens at
    16:45 and does not accept orders again until `trading_resumes_et` (18:00). The
    16:45-18:00 window is closed to the trader entirely. That gap nearly promoted an
    FOMC entry hour whose whole advantage over 18:00 turned out to be the untradeable
    17:00-18:00 CME maintenance gap -- exactly the error that killed the fake
    "Asia drift" earlier in this project.
    """
    if rules.get("flatten_et") is None:
        # No daily auto-liquidation at all. Only genuine market closure constrains the
        # hold, so any window this project can express is legal here.
        return True, "firm imposes no mandatory daily flatten"
    fh, fm = (int(x) for x in rules["flatten_et"].split(":"))
    flat = fh * 60 + fm
    rh, rm = (int(x) for x in rules["resumes_et"].split(":"))
    resume = rh * 60 + rm
    eh, em = (int(x) for x in entry_et.split(":"))
    xh, xm = (int(x) for x in exit_et.split(":"))
    e, x = eh * 60 + em, xh * 60 + xm
    if flat < e < resume:
        return False, (f"entry {entry_et} falls in the {rules['flatten_et']}-"
                       f"{rules['resumes_et']} ET closed window -- no orders accepted")
    if spans_days == 0:
        return (True, "intraday, no flatten crossed") if not (e < flat < x) else \
            (False, f"crosses the {rules['flatten_et']} ET flatten")
    # overnight: legal only if entry is after the flatten and exit is before the next one
    if e >= resume and x < flat:
        return True, "opens after trading resumes, closes before the next flatten"
    return False, (f"held across a {rules['flatten_et']} ET mandatory flatten "
                   "-- the position is force-closed before the window completes")
