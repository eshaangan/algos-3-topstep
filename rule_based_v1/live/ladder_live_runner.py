"""UNIFIED live daemon for the validated book — one Rithmic order-plant session.

Strategies (all pre-registered):
  WK   weekend_hold_v1 : long Sun 18:00 -> Mon 16:00 ET          (ladder rung sizing)
  EO   euro_open_v1    : long 2:00 -> 5:00 ET weeknights IF prior RTH day down & high-range
  FOMC fomc_drift_v1   : long 18:00 day-before -> 13:55 decision day (verified Fed calendar)
                         18:00 is the earliest LEGAL entry: 14:00/16:00 cross the
                         16:45 flatten, and 16:45-18:00 accepts no orders.

Guardrails: hard size ladder (banked-PnL cushion), catastrophic verified stop on
every position, fill-timeout reconciliation, flatten verification sweep, holiday
guard, KILL file, watchdog, atomic state, ntfy phone alerts. DRY unless --live.
"""
from __future__ import annotations
import asyncio, glob, json, math, os, time, urllib.request, uuid
from datetime import date, datetime, timedelta, timezone
import pandas as pd

ET = "America/New_York"


def _secrets():
    """Every value that must never reach a log line or a push notification.

    Redaction used to cover RITHMIC_PASSWORD only, while the account id was logged in
    clear and ntfy.sh is a PUBLIC topic -- anyone who learns the topic string could read
    position size, PnL and balance.
    """
    return [v for v in (os.environ.get(k, "") for k in
                        ("RITHMIC_PASSWORD", "RITHMIC_USERNAME", "LUCID_ACCOUNT_ID",
                         "NTFY_TOPIC")) if v and len(v) >= 4]


class _RedactStream:
    """Wrap a text stream; redact every credential from every write. This is the
    guaranteed backstop: the supervisor captures our stdout/stderr into runner.log,
    so scrubbing the streams catches leaks from ANY source (rithmic.plant.* child
    loggers, tracebacks, prints) regardless of logging configuration.

    Carries a one-write tail so a secret split ACROSS two write() calls -- which is
    exactly what logging does when it emits a record in pieces -- is still caught."""
    def __init__(self, wrapped, secrets):
        self._w = wrapped; self._s = list(secrets); self._tail = ""
    def write(self, m):
        try:
            joined = self._tail + m
            for sec in self._s:
                if sec in joined:
                    joined = joined.replace(sec, "***")
                    m = joined[len(self._tail):]
            self._tail = m[-64:] if m else ""
        except Exception:
            pass
        return self._w.write(m)
    def flush(self):
        try: return self._w.flush()
        except Exception: pass
    def __getattr__(self, n):
        return getattr(self._w, n)


def _wrap_streams():
    import sys as _sys
    secs = _secrets()
    if secs and not isinstance(_sys.stdout, _RedactStream):
        _sys.stdout = _RedactStream(_sys.stdout, secs)
        _sys.stderr = _RedactStream(_sys.stderr, secs)


_wrap_streams()   # install at import, before any logging handler is constructed
SYMBOL, EXCHANGE = "MNQ", "CME"
PV, COMM, TICK = 2.0, 0.62, 0.25


def _load_rules():
    """Session/limit constants from configs/account_rules.yaml -- the single source of
    truth. These were hardcoded here AND stated in the yaml, which is how the project
    previously ran months of work against a wrong MLL. Falls back only if the loader
    is unavailable (the yaml itself is never second-guessed)."""
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
        from rule_based_v1.validation.tzguard import load_account_rules
        r = load_account_rules()
        fh, fm = (int(x) for x in r["flatten_et"].split(":"))
        rh, rm = (int(x) for x in r["resumes_et"].split(":"))
        return (float(r["max_loss_limit"]), float(r["profit_target"]),
                float(r["consistency"]), fh * 60 + fm, rh * 60 + rm, bool(r["verified"]))
    except Exception as e:                       # pragma: no cover - config must load
        raise RuntimeError(f"cannot load account_rules.yaml: {e}") from e


MLL_USD, PROFIT_TARGET, CONSISTENCY_MAX, FLATTEN_HM, RESUME_HM, RULES_VERIFIED = _load_rules()
# Self-flatten margin: close this many minutes BEFORE the firm's mandatory flatten, so
# an exit is ours and priced, never the firm's auto-liquidation at an arbitrary print.
FLATTEN_MARGIN_MIN = 5
# How often to re-confirm that the catastrophe stop is still resting. These holds run
# up to 22h; verifying once at entry leaves the rest of that unmonitored.
RECHECK_SEC = 120
# Loop interval and the watchdog's patience for one iteration.
LOOP_SEC = 15
WATCHDOG_SEC = 300
TAIL_BUDGET = 0.60        # worst single trade must stay under 60% of the cushion
# MC-DERIVED SIZE CAPS (research_disaster_stop / research_firm_choice_mc, $600 stop,
# 104wk, flex_100k). The tail budget alone is NECESSARY BUT NOT SUFFICIENT: with the stop
# in place a 2-micro weekend passes the 60% tail test, yet it drops P(pass) from 96.2% to
# 80.9% -- below the 85% bar -- because the weekend leg's variance, not its worst trade,
# is what busts the account. Measured grid:
#     WK1/FOMC2 96.2%/35wk   WK1/FOMC3 93.9%/32wk   WK1/FOMC4 90.0%/30wk
#     WK2/FOMC2 80.9%/17wk   WK2/FOMC3 80.2%/16wk   WK2/FOMC4 78.5%/16wk
# So WK stays at 1 until the account is nearly made, and the event leg carries the size.
def size_cap(kind, banked):
    if kind == "WK":
        return 2 if banked >= 4500 else 1
    return 3 if banked >= 4500 else 2


# ── WHAT IS AUTHORISED TO TRADE LIVE, AND BY WHAT ────────────────────────────────
# validation/preregister.yaml is the authority. Each leg's live size is capped here by
# what its OWN registration permits, because the runner previously sized every leg from
# the MC grid alone and traded two legs the registration does not authorise at all.
#
#   WK  weekend_hold_v1  deployment.phase1 = "paper + optional 1-micro live shakedown".
#                        go_scale (2-3 micros) requires 8 forward weekends with mean > 0
#                        and >= 5/8 positive. Forward record as of 2026-09-06: 5 weekends,
#                        mean -$286, t=-0.97. The gate is NOT met, so 1 micro is the cap.
#                        (kill = 8 weekends mean < 0; at 5 it has not fired either.)
#   FOMC fomc_drift_v1   The registered spec is a 14:00 entry, which is ILLEGAL here. The
#                        legal 18:00 replacement FAILED its own promotion criteria in
#                        vps_legal_fomc_window_VERDICT.md: minus-control t=+2.29 vs a 2.5
#                        bar, Bonferroni needs 2.90, and the ES dev half is -$1.87
#                        (t=-0.17), which that file calls a registered kill. It was kept
#                        as "a portfolio leg at reduced size" -- a disposition decided
#                        after seeing the result, not a pre-registered path.
#   EO  euro_open_v1     status = "BELOW GO (t<2). No tuning permitted. Forward nights
#                        accrue via paper runner; promote if combined t>=2.5." It is at
#                        t=1.84 and failed cross-instrument on ES (t=-0.31). It has never
#                        been promoted, and it is not in any portfolio MC -- so its risk
#                        against the $3,000 MLL is entirely unbudgeted.
#
# Set a leg to 0 to keep it paper-only. Raising any of these is a research decision that
# belongs in preregister.yaml FIRST, not a config tweak here.
AUTHORISED_MAX = {
    "WK": 1,      # phase1 shakedown size; go_scale not met
    "FOMC": 0,    # failed its own promotion criteria at the legal 18:00 entry
    "EO": 0,      # never promoted; watchlist only
}
# VALIDATED DISASTER STOP (2026-09-05, vps_disaster_stop_sizing_VERDICT.md).
# $600/micro keeps 93% of the weekend premium while cutting the worst trade 46%
# (-$1,117 -> -$603/micro), raising P(pass) 92.0->94.4% and decay-robustness
# 70.9->73.2%. It REPLACES the old 450pt "catastrophe" stop, which was so wide it
# almost never bound and left the worst trade at 37% of the MLL.
STOP_USD_PER_MICRO = 600.0
CAT_STOP_PTS = STOP_USD_PER_MICRO / PV            # 300 pts on MNQ

# Worst historical single-trade loss per micro WITH the stop in place, measured with
# gap-through fills (stop fills at min(stop, bar open)) over n=202 weekends and n=47
# FOMC events: -$603. Carrying 8% headroom for a Sunday gap worse than any in sample.
#
# WAS: "WK worst MAE ~ -$2,092/micro" and "EO/FOMC ~ -$416/micro". The FOMC figure was
# 2x OPTIMISTIC (measured worst is -$756/micro unstopped), so every ladder rung carried
# twice its intended risk. Both are now superseded by the stop, which BOUNDS the loss
# instead of estimating it.
WORST_STOPPED_PER_MICRO = 650.0

# Scheduled FOMC announcement dates, verified against federalreserve.gov's published
# calendar (fomccalendars.htm) on 2026-09-06. Entry is derived as the prior day, never
# maintained as a second hand-edited list -- the old code kept two lists that could
# silently disagree. The old calendar ENDED 2026-12-08.
FOMC_ANNOUNCE = ["2026-09-16", "2026-10-28", "2026-12-09",
                 "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09",
                 "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08"]


def _fomc_tables(announce):
    """entry-date list and entry->announce map, derived from one source list.

    Every scheduled FOMC is a Tue-Wed two-day meeting, so the session before the
    announcement is the previous calendar day. Asserted rather than assumed: a
    Monday announcement would mean the prior session is Friday and this derivation
    would silently book a weekend-spanning hold.
    """
    entries, exits = [], {}
    for a in announce:
        d = date.fromisoformat(a)
        e = d - timedelta(days=1)
        if e.weekday() > 4:
            raise AssertionError(f"FOMC {a}: prior day {e} is a weekend; entry date "
                                 "must be the previous SESSION -- fix the calendar")
        entries.append(e.isoformat())
        exits[e.isoformat()] = a
    return entries, exits


FOMC_ENTRY, FOMC_EXIT = _fomc_tables(FOMC_ANNOUNCE)

# CME equity-index full closures (no RTH). Extended through 2027 because FOMC_ANNOUNCE
# now runs to 2027-12-08: the two calendars have to cover the same span or the runner
# opens a weekend hold into a Monday with no RTH session.
HOLIDAYS = {
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas
    "2027-01-01",  # New Year's Day
    "2027-01-18",  # MLK
    "2027-02-15",  # Presidents' Day
    "2027-03-26",  # Good Friday
    "2027-05-31",  # Memorial Day
    "2027-07-05",  # Independence Day (observed)
    "2027-09-06",  # Labor Day
    "2027-11-25",  # Thanksgiving
    "2027-12-24",  # Christmas (observed)
}


def _no_entry_sundays(holidays):
    """Sundays whose MONDAY is a holiday -- derived, never hand-maintained.

    WAS a second hand-edited set holding exactly one date. Two hand-maintained
    calendars that must agree is the same failure mode _fomc_tables() was written to
    remove; a missing entry here opens a weekend hold into a closed Monday.
    """
    out = set()
    for h in holidays:
        d = date.fromisoformat(h)
        if d.weekday() == 0:                      # Monday holiday -> block the Sunday
            out.add((d - timedelta(days=1)).isoformat())
    return out


NO_ENTRY_SUNDAYS = _no_entry_sundays(HOLIDAYS)

# Every window this runner can OPEN, as (entry_et, exit_et, spans_days), matching the
# minute bounds in schedule(). Checked against tzguard at import so an edit to those
# bounds cannot quietly produce an illegal window: the entry must not fall in the
# 16:45-18:00 closed period, and the hold must not cross the 16:45 mandatory flatten.
TRADE_WINDOWS = {
    "WK":   [("18:00", "16:00", 1), ("18:15", "16:00", 1)],   # Sun 18:00-18:15 -> Mon
    "FOMC": [("18:00", "13:55", 1), ("18:10", "13:55", 1)],   # prior session -> decision
    "EO":   [("02:00", "05:00", 0), ("02:10", "05:00", 0)],   # weeknights, one session
}


def assert_windows_legal():
    """Machine-check every openable window against the firm's session rules."""
    from rule_based_v1.validation.tzguard import check_window_legal, load_account_rules
    rules = load_account_rules()
    bad = []
    for kind, wins in TRADE_WINDOWS.items():
        for entry, exit_, spans in wins:
            ok, why = check_window_legal(entry, exit_, spans, rules)
            if not ok:
                bad.append(f"{kind} {entry}->{exit_} (+{spans}d): {why}")
    if bad:
        raise AssertionError("ILLEGAL TRADE WINDOW(S):\n  " + "\n  ".join(bad))
    return True


assert_windows_legal()

# --- contract rollover -------------------------------------------------------------
# MNQ is quarterly (H/M/U/Z), expiring the third Friday of the contract month. The CME
# equity-index roll is 8 days before expiry (the Thursday of the week BEFORE expiry
# week) -- that is where volume actually moves, not the day before expiry.
# WAS: a bare `os.environ.get("FADE_CONTRACT", "MNQU6")`. MNQU6 expired 2026-09-18 and
# stopped being front month on 2026-09-10 -- i.e. it was already stale for every date
# this runner can trade. There was no rollover and no expiry guard at all.
_QMONTH = {3: "H", 6: "M", 9: "U", 12: "Z"}


def _third_friday(y, m):
    d = date(y, m, 1)
    d += timedelta(days=(4 - d.weekday()) % 7)     # first Friday
    return d + timedelta(days=14)


ROLL_LEAD_DAYS = 8


def front_month(today):
    """Front-month MNQ code for `today` (a date), e.g. MNQZ6.

    Returns (code, roll_date) so callers can refuse to OPEN a hold that would still be
    running when liquidity leaves the contract.
    """
    y = today.year
    for m in (3, 6, 9, 12):
        roll = _third_friday(y, m) - timedelta(days=ROLL_LEAD_DAYS)
        if today < roll:
            return f"MNQ{_QMONTH[m]}{y % 10}", roll
    roll = _third_friday(y + 1, 3) - timedelta(days=ROLL_LEAD_DAYS)
    return f"MNQ{_QMONTH[3]}{(y + 1) % 10}", roll


NTFY = os.environ.get("NTFY_TOPIC", "")

def _scrub(m):
    if not isinstance(m, str):
        return m
    for sec in _secrets():
        m = m.replace(sec, "***")
    return m

def log(m): print(f"[{datetime.now(timezone.utc):%m-%d %H:%M:%S}Z] {_scrub(str(m))}", flush=True)

def _install_redaction():
    """Handler-level redaction: filters on HANDLERS see records propagated from child
    loggers (rithmic.plant.*), unlike filters on loggers. Renders msg%args first so
    the secret is caught in either. Belt-and-suspenders with _wrap_streams()."""
    import logging
    _wrap_streams()
    sec = os.environ.get("RITHMIC_PASSWORD", "")
    if not sec:
        return
    class _R(logging.Filter):
        def filter(self, rec):
            try:
                msg = rec.getMessage()
                if sec in msg:
                    rec.msg = msg.replace(sec, "***"); rec.args = ()
            except Exception:
                pass
            return True
    filt = _R()
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
    for h in root.handlers:
        h.addFilter(filt)
    for nm in list(logging.root.manager.loggerDict):
        for h in getattr(logging.getLogger(nm), "handlers", []):
            h.addFilter(filt)
    logging.getLogger("rithmic").setLevel(logging.WARNING)
    logging.getLogger("async_rithmic").setLevel(logging.WARNING)
def push(m):
    m = _scrub(str(m))
    if not NTFY: return
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"https://ntfy.sh/{NTFY}", data=m.encode(), method="POST"), timeout=5)
    except Exception: pass

def cushion_for(banked, peak_banked=None):
    """Distance from equity to the EOD-trailing floor.

    The floor trails the equity PEAK by the MLL but LOCKS at the starting balance
    (account_rules.yaml::drawdown.locks_at_starting_balance).

        floor  = min(start, start + peak_banked - MLL)
        equity = start + banked
        cushion = equity - floor = banked - min(0, peak_banked - MLL)

    WAS: `max(MLL_USD, banked)`, which read CURRENT banked where the floor tracks the
    PEAK. That is correct only while the account sits at its high-water mark. After a
    peak-then-drawdown it overstates the cushion by up to 30x -- peak $2,900 / now $0
    is a true cushion of $100, and the old form returned $3,000. Below the lock point
    every dollar given back is a dollar of cushion gone; the old form never shrank.
    """
    b = float(banked)
    pk = b if peak_banked is None else max(float(peak_banked), b)
    return max(0.0, b - min(0.0, pk - MLL_USD))


def ladder_size(kind, banked, cushion=None, peak_banked=None, live=False):
    """Largest size whose worst STOPPED loss stays inside TAIL_BUDGET of the cushion.

    Derived from measured constants rather than hand-written rungs. The old version
    hardcoded rungs against a FOMC worst-MAE that was 2x optimistic, so each rung
    carried twice its intended risk; computing the rung from the constant means fixing
    the constant fixes every rung at once.

    `cushion` is the BROKER-reported cushion when one is available -- callers must
    pass it. It used to be ignored at the call site, so sizing ran off the internal
    estimate while only a separate coarse gate saw the truth.

    Returns 0 (not 1) when even one micro breaches the budget. A floor of 1 meant a
    nearly-busted account still took a full-size position.
    """
    c = cushion_for(banked, peak_banked) if cushion is None else float(cushion)
    n = int((TAIL_BUDGET * c) // WORST_STOPPED_PER_MICRO)
    n = min(n, size_cap(kind, float(banked)))
    if live:
        n = min(n, AUTHORISED_MAX.get(kind, 0))    # pre-registration cap, live only
    return max(0, n)

class Daemon:
    def __init__(self, out, live, now_fn=None, quote_fn=None):
        self.out, self.live = out, live
        os.makedirs(out, exist_ok=True)
        self.state_p = os.path.join(out, "state.json")
        self.ev_p = os.path.join(out, "events.jsonl")
        self.kill_p = os.path.join(out, "KILL")
        self.st = json.load(open(self.state_p)) if os.path.exists(self.state_p) else \
            {"open": None, "done": [], "banked": 0.0}
        # peak_banked drives the EOD-trailing floor; seed it from banked for states
        # written before it existed. day_pnl/best_day back the 50% consistency rule.
        self.st.setdefault("peak_banked", float(self.st.get("banked", 0.0)))
        self.st.setdefault("day_pnl", {})
        for _f in (self.state_p, self.ev_p):
            try:
                if os.path.exists(_f): os.chmod(_f, 0o600)
            except OSError: pass
        self.client = None; self.acct = None; self.contract = None
        self.pos_qty = None; self.pos_qty_ts = 0.0   # never implicitly absent
        self.now = now_fn or (lambda: pd.Timestamp.now(tz=ET))
        self.quote = quote_fn or self._stream_quote
        self.beat = time.time()
        self._stop = False; self._halt_logged = False
        self.roll_date = date.max

    def jlog(self, **r):
        r["t"] = str(pd.Timestamp.now(tz="UTC"))
        with open(self.ev_p, "a") as f:
            f.write(json.dumps(r, default=str) + "\n")

    def save(self):
        """Atomic + durable. os.replace() alone is atomic but the bytes can still be
        in page cache when a VPS loses power, which is how a half-written position
        record appears on restart."""
        tmp = self.state_p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.st, f, default=str)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, self.state_p)

    def book_pnl(self, pnl, kind, key, why):
        """Single place where realised PnL enters state.

        Everything that closes a position must come through here: banked, the
        peak-banked high-water mark that sets the trailing floor, the per-day totals
        the consistency rule needs, and the done-key all move together or not at all.
        Previously each exit path updated a different subset.
        """
        pnl = float(pnl)
        self.st["banked"] = float(self.st["banked"]) + pnl
        self.st["peak_banked"] = max(float(self.st.get("peak_banked", 0.0)),
                                     float(self.st["banked"]))
        d = str(self.now().date())
        self.st["day_pnl"][d] = round(float(self.st["day_pnl"].get(d, 0.0)) + pnl, 2)
        if key is not None and key not in self.st["done"]:
            self.st["done"].append(key)
        self.st["open"] = None
        self.save()
        self.jlog(ev="closed", kind=kind, key=key, why=why, pnl=round(pnl, 2),
                  banked=round(self.st["banked"], 2),
                  peak=round(self.st["peak_banked"], 2), consistency=self.consistency())
        return pnl

    def consistency(self):
        """Largest profitable day as a fraction of total profit (the 50% rule).

        account_rules.yaml::consistency.max_best_day_fraction is enforced at the moment
        of pass, so a book of ~1 trade/week can hit the $6,000 target and still fail.
        Nothing in this runner tracked it at all before; it is reported on every close
        and gates entries once the account is close enough to pass for it to bind.
        """
        days = {d: v for d, v in self.st.get("day_pnl", {}).items() if v > 0}
        total = sum(days.values())
        if total <= 0:
            return None
        return round(max(days.values()) / total, 4)

    def _stream_quote(self):
        """Reconstruct a two-sided mid from the recorder's ONE-SIDED BBO stream:
        each line carries a bid update OR an ask update (other side NaN). We scan the
        tail newest->oldest and take the most recent finite bid and finite ask.

        Files are matched on the RESOLVED CONTRACT, falling back to the generic symbol
        only when the recorder does not name the month. The glob used to be an
        unconditional `stream_MNQ_*`, so across a roll the reference price could come
        from a different contract than the one being ordered -- a 40-80pt basis error
        feeding a 300pt stop.
        """
        raw = os.environ.get("RAW_DIR", "")
        fs = sorted(glob.glob(os.path.join(raw, f"stream_{self.contract}_*.csv"))) \
            if self.contract else []
        if not fs:
            fs = sorted(glob.glob(os.path.join(raw, "stream_MNQ_*.csv")))
        if not fs: return None, 1e9
        with open(fs[-1], "rb") as f:
            try: f.seek(-65536, 2)
            except OSError: f.seek(0)
            lines = f.read().decode(errors="ignore").strip().splitlines()
        bid = ask = newest_ns = None
        for line in reversed(lines):
            p = line.split(",")
            if len(p) != 5: continue
            try:
                ns = float(p[0]); b = float(p[1]); a = float(p[3])
            except ValueError:
                continue
            if newest_ns is None: newest_ns = ns
            if bid is None and math.isfinite(b): bid = b
            if ask is None and math.isfinite(a): ask = a
            if bid is not None and ask is not None: break
        if bid is None or ask is None or newest_ns is None: return None, 1e9
        mid = (bid + ask) / 2
        if mid <= 0: return None, 1e9
        return mid, time.time() - newest_ns / 1e9

    async def connect(self):
        _install_redaction()
        from async_rithmic import RithmicClient, ReconnectionSettings, SysInfraType
        self.client = RithmicClient(
            user=os.environ["RITHMIC_USERNAME"], password=os.environ["RITHMIC_PASSWORD"],
            system_name=os.environ.get("RITHMIC_SYSTEM_NAME", "LucidTrading"),
            app_name=os.environ.get("RITHMIC_APP_NAME", "x") + ":ladder", app_version="1.0.0",
            url=os.environ.get("RITHMIC_GATEWAY_URI", "wss://rprotocol.rithmic.com:443"),
            reconnection_settings=ReconnectionSettings(max_retries=None,
                backoff_type="exponential", interval=3, max_delay=120))
        await self.client.connect(plants=[SysInfraType.ORDER_PLANT, SysInfraType.PNL_PLANT])
        accounts = await self.client.plants["order"].list_accounts()
        want = os.environ.get("LUCID_ACCOUNT_ID")
        match = [a for a in accounts if not want or str(a.account_id) == want]
        if not match:
            raise RuntimeError(f"account {want} not found in {[a.account_id for a in accounts]}")
        self.acct = match[0].account_id
        self.contract, self.roll_date = self.resolve_contract()
        self.pos_qty = None           # live position per PNL plant
        self.client.on_exchange_order_notification += self._note
        self.client.on_instrument_pnl_update += self._pnl_note
        self.client.on_account_pnl_update += self._acct_note
        # PNL SUBSCRIPTION. This lives on the PNL PLANT, not on the client -- the old
        # `self.client.subscribe_to_pnl_updates(...)` raised AttributeError (RithmicClient
        # has no such method and no __getattr__), both branches swallowed it, and the
        # daemon ran believing it was subscribed. With no PNL feed `pos_qty` stays None
        # forever, so every flatten() ends in an unconfirmed HALT, and `acct_balance` is
        # never set, so _broker_cushion() returns None and blocks EVERY live entry.
        # The subscription is load-bearing for both safety paths: failure is fatal here,
        # not a log line. The library re-subscribes itself on reconnect (PnlPlant._login).
        await self.client.plants["pnl"].subscribe_to_pnl_updates()
        log(f"connected acct={self.acct} {self.contract} (rolls {self.roll_date}) "
            f"mode={'LIVE' if self.live else 'DRY'}")
        push(f"ladder daemon up ({'LIVE' if self.live else 'DRY'})")

    def resolve_contract(self):
        """(code, roll_date) for today, with an explicit override.

        An override is honoured but must still name a contract that is currently front
        month -- a stale FADE_CONTRACT pinned in the environment is exactly how MNQU6
        survived past its 2026-09-10 roll as the hardcoded default.
        """
        today = self.now().date()
        code, roll = front_month(today)
        override = os.environ.get("LADDER_CONTRACT") or os.environ.get("FADE_CONTRACT")
        if override and override != code:
            raise RuntimeError(
                f"contract override {override!r} is not the front month for {today} "
                f"(expected {code}, rolls {roll}). Unset LADDER_CONTRACT/FADE_CONTRACT "
                "to use the computed front month.")
        return code, roll

    async def _note(self, n):
        from async_rithmic import ExchangeOrderNotificationType as NT
        if getattr(n, "notify_type", None) != NT.FILL: return
        px, side = getattr(n, "fill_price", None), getattr(n, "transaction_type", None)
        sym = getattr(n, "symbol", ""); tag = str(getattr(n, "user_tag", "") or "")
        acct = getattr(n, "account_id", None)
        qty = int(getattr(n, "fill_size", 0) or 0)
        self.jlog(ev="fill", px=px, side=side, tag=tag, sym=sym, qty=qty, acct=acct)
        ot = self.st.get("open")
        if not (ot and px): return
        # STRICT attribution: our contract, our account (if reported), our order tags only
        if sym != self.contract: return
        if acct and self.acct and str(acct) != str(self.acct): return
        ours = tag.startswith(ot["oid"]) or tag == ""      # exit_position fills carry no tag
        if not ours: return
        if side == 1 and tag == ot["oid"]:
            filled = min(self._accum(ot, "filled", qty or ot["qty"]), ot["qty"])
            ot["filled_qty"] = filled
            # VWAP, not first-slice. protect() prices the stop off this, and on a
            # multi-slice fill the first print is not what the position cost.
            ot["fill_notional"] = ot.get("fill_notional", 0.0) + float(px) * (qty or ot["qty"])
            ot["fill"] = ot["fill_notional"] / max(filled, 1)
            self.save()
            log(f"ENTRY FILL {filled}/{ot['qty']} @ {px} (avg {ot['fill']:.2f})")
            if filled >= ot["qty"]: push(f"{ot['kind']} entry filled @ {ot['fill']:.2f}")
        elif side == 2:
            # A SELL we did not ask for is normally an alarm -- but the firm's mandatory
            # 16:45 flatten arrives exactly like this (no tag, no `closing` flag), and
            # rejecting it left the position booked open forever and `banked` stale,
            # which then feeds size_cap. Attribute it, and say which it was.
            forced = not (ot.get("closing") or tag.endswith("-stp"))
            if forced and not self._plausible_firm_flatten():
                log(f"UNKNOWN SELL fill tag={tag!r} — NOT attributing"); push("CRITICAL unknown SELL fill — check account")
                self.jlog(ev="unknown_sell_fill", tag=tag, px=px); return
            if forced:
                log("SELL with no tag near the 16:45 flatten — booking as firm auto-flatten")
                self.jlog(ev="firm_flatten_fill", tag=tag, px=px)
            fqty = ot.get("filled_qty", ot["qty"])
            sold = min(self._accum(ot, "sold", qty or fqty), fqty)
            ot["sold_qty"] = sold; self.save()
            if sold < fqty: return                          # partial exit; wait for rest
            pnl = (float(px) - (ot.get("fill") or ot["ref"]))*PV*fqty - 2*COMM*fqty
            self.book_pnl(pnl, ot["kind"], self.done_key(ot["kind"], ot["key"]),
                          "firm_flatten" if forced else "stop" if tag.endswith("-stp") else "exit")
            log(f"CLOSED {ot['kind']} @ {px} pnl=${pnl:+.2f} banked=${self.st['banked']:+.2f}")
            push(f"{ot['kind']} closed {pnl:+.2f} (banked {self.st['banked']:+.0f})")
            asyncio.create_task(self._cancel_all())

    @staticmethod
    def _accum(ot, field, size):
        """Running filled/sold total, tolerant of CUMULATIVE fill_size reports.

        Rithmic's fill_size semantics (incremental vs running total) are not guaranteed
        across gateways. Blind addition double-counts if they are cumulative, which
        overstates filled_qty and makes protect() place an OVERSIZED stop -- leaving a
        residual naked short when it triggers.

        max(running sum, largest single report) is correct under BOTH conventions once
        the caller clamps to the ordered quantity:
            incremental 1,1,1 -> sum 3, max 1 -> 3
            cumulative  1,2,3 -> sum 6, max 3 -> 6, clamped to 3
        """
        size = int(size)
        ot[f"{field}_sum"] = int(ot.get(f"{field}_sum", 0)) + size
        ot[f"{field}_max"] = max(int(ot.get(f"{field}_max", 0)), size)
        return max(ot[f"{field}_sum"], ot[f"{field}_max"])

    def _plausible_firm_flatten(self):
        """True in the ET window where the firm's mandatory auto-liquidation lands."""
        now = self.now()
        hm = now.hour * 60 + now.minute
        return now.weekday() <= 4 and FLATTEN_HM <= hm < RESUME_HM

    async def _pnl_note(self, n):
        try:
            if str(getattr(n, "account_id", "") or "") not in ("", str(self.acct)):
                return
            if getattr(n, "symbol", "") == self.contract:
                q = getattr(n, "open_position_quantity", None)
                if q is None: q = getattr(n, "fill_buy_qty", 0) - getattr(n, "fill_sell_qty", 0)
                self.pos_qty = int(q); self.pos_qty_ts = time.time()
        except Exception: pass

    async def _acct_note(self, n):
        if str(getattr(n, "account_id", "") or "") not in ("", str(self.acct)):
            return
        for f in ("account_balance", "cash_on_hand", "net_liquidating_value", "current_balance"):
            v = getattr(n, f, None)
            if v:
                try: self.acct_balance = float(v); return
                except (TypeError, ValueError): pass

    async def _call(self, what, coro, timeout=20):
        """Every broker await goes through here.

        None of them had a timeout. A hung await -- the realistic outcome of a Rithmic
        reconnect storm -- stalled the whole 15s loop indefinitely with a position open
        and no alert, because the advertised watchdog did not exist either.
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            log(f"{what}: TIMEOUT after {timeout}s"); self.jlog(ev="broker_timeout", call=what)
            raise
        except Exception as e:
            log(f"{what} err {e}"); self.jlog(ev="broker_error", call=what, err=str(e))
            raise

    async def _cancel_all(self):
        try:
            await self._call("cancel_all",
                             self.client.plants["order"].cancel_all_orders(account_id=self.acct))
        except Exception:
            pass

    async def _broker_cushion(self):
        """Broker-reported distance to the auto-liquidation floor, or None.

        None means "unknown" and blocks entry. Getting that distinction right matters:
        `min_account_balance` is a proto3 field WITHOUT presence, so getattr() returns
        0.0 when the gateway never populated it -- never None. The old `floor is None`
        check therefore could not fire, and an unset floor silently read as a ~$100,000
        cushion, turning the last-line-of-defence gate into a no-op. Sanity-bound the
        pair instead of trusting a sentinel that cannot occur.
        """
        try:
            rms = await self._call("get_account_rms",
                                   self.client.plants["order"].get_account_rms())
            recs = list(rms) if isinstance(rms, (list, tuple)) else [rms]
            rec = next((r for r in recs
                        if str(getattr(r, "account_id", "")) == str(self.acct)), None)
            if rec is None:
                self.jlog(ev="broker_cushion_no_account_record", n=len(recs))
                return None
            floor = getattr(rec, "min_account_balance", None)
            bal = getattr(self, "acct_balance", None)   # from account-filtered PNL plant
            if bal is None or floor is None:
                self.jlog(ev="broker_cushion_fields_missing",
                          have_floor=floor is not None, have_bal=bal is not None)
                return None
            floor, bal = float(floor), float(bal)
            # An unpopulated proto field reads as exactly 0.0; so does a real floor of
            # zero, which no funded evaluation account has. Refuse both.
            if floor <= 0.0 or bal <= 0.0:
                log(f"broker cushion unusable (floor={floor} bal={bal}) — treating as unknown")
                self.jlog(ev="broker_cushion_unpopulated", floor=floor, bal=bal)
                return None
            c = bal - floor
            # Algebraic ceiling: with floor = min(start, start + peak - MLL),
            # cushion = banked - min(0, peak - MLL) <= max(MLL, banked). Anything above
            # that is not this account's floor -- e.g. an unset field read as 0.0, which
            # would present the full ~$100,000 balance as cushion.
            ceiling = max(MLL_USD, float(self.st["banked"])) * 1.05 + 100.0
            if not (0.0 <= c <= ceiling):
                log(f"broker cushion {c:.0f} outside [0, {ceiling:.0f}] — treating as unknown")
                self.jlog(ev="broker_cushion_out_of_range", cushion=c, bal=bal, floor=floor)
                return None
            self.jlog(ev="broker_cushion", cushion=c, bal=bal, floor=floor)
            return c
        except Exception:
            return None

    async def _list_orders(self):
        try:
            return list(await self._call(
                "list_orders", self.client.plants["order"].list_orders(account_id=self.acct)))
        except Exception:
            return None

    def _our_working_tags(self, orders):
        """Tags of orders on our contract that are NOT in a terminal state."""
        out = []
        for o in orders or []:
            if getattr(o, "symbol", self.contract) not in ("", self.contract):
                continue
            status = str(getattr(o, "status", "") or "").lower()
            if any(k in status for k in ("complete", "cancel", "fill", "reject", "done")):
                continue
            out.append(str(getattr(o, "user_tag", "") or ""))
        return out

    @staticmethod
    def done_key(kind, key):
        """The ONE de-dup key for a scheduled trade.

        FOMC used to test `f"F{d}"` in schedule() while every completion path appended
        the bare `d`, so the test never matched: after any flatten or block inside the
        18:00-18:10 window the daemon re-entered on the very next 15s tick, repeatedly.
        WK and EO were self-consistent by luck. One function now produces the key for
        both the test and the write, so they cannot drift apart again.
        """
        return f"{kind}:{key}"

    def is_done(self, kind, key):
        return self.done_key(kind, key) in self.st["done"]

    def mark_done(self, kind, key):
        k = self.done_key(kind, key)
        if k not in self.st["done"]:
            self.st["done"].append(k); self.save()

    @staticmethod
    def _exit_date(kind, key):
        """Calendar date on which this window closes.

        WK's key IS the Monday and EO's is the same session, so both are the key itself.
        An FOMC key must be in the derived table; a miss means the calendar and the
        scheduler disagree, which is a refuse-to-trade condition, not a crash.
        """
        if kind in ("WK", "EO"):
            return date.fromisoformat(key)
        exit_iso = FOMC_EXIT.get(key)
        if exit_iso is None:
            raise KeyError(f"FOMC key {key!r} is not in the derived calendar")
        return date.fromisoformat(exit_iso)

    def consistency_block(self, kind):
        """Refuse a trade that would make the 50% best-day rule unpassable.

        account_rules.yaml::consistency enforces `best_day <= 50% of total profit` at
        the moment of pass. It only binds near the target: below that there is room for
        later days to dilute a big one. Once banked is within one good trade of
        PROFIT_TARGET, a single outsized day is how this account fails AFTER making the
        money -- which nothing in this runner previously even measured.
        """
        if self.st["banked"] < PROFIT_TARGET * 0.6:
            return None                                    # far from target; cannot bind
        c = self.consistency()
        if c is not None and c > CONSISTENCY_MAX:
            return (f"consistency {c:.0%} > {CONSISTENCY_MAX:.0%} with banked "
                    f"${self.st['banked']:.0f} — need more spread days, not more size")
        return None

    async def _reconcile_after_submit_failure(self, oid, kind, key):
        """Did the order that errored actually reach the exchange?

        Three outcomes: it is working/filled (keep the position, let the protection
        lifecycle run), it is provably absent (clear state, allow a retry), or we cannot
        tell (halt -- never guess, because guessing wrong doubles the position).
        """
        await asyncio.sleep(3)
        orders = await self._list_orders()
        if orders is None:
            log("CRITICAL: submit failed AND order list unavailable — HALTING")
            push("CRITICAL: submit failed, cannot verify — daemon halted, CHECK ACCOUNT")
            self.jlog(ev="submit_fail_unverifiable_halt", oid=oid)
            self.st["halted"] = True; self.save(); return
        seen = [o for o in orders if str(getattr(o, "user_tag", "") or "").startswith(oid)]
        if seen:
            log(f"submit error but order {oid} IS live — keeping position, protecting it")
            self.jlog(ev="submit_fail_but_live", oid=oid)
            return                                          # st["open"] stays; loop protects it
        fresh = self.pos_qty_ts >= self.st["open"]["entered_at"] if self.st.get("open") else False
        if self.pos_qty in (0, None) and not fresh:
            log("CRITICAL: submit failed, no order found, position unconfirmed — HALTING")
            push("CRITICAL: submit failed, position unconfirmed — daemon halted")
            self.jlog(ev="submit_fail_unconfirmed_halt", oid=oid)
            self.st["halted"] = True; self.save(); return
        if self.pos_qty == 0 and fresh:
            log(f"submit {oid} provably never reached the exchange — clearing for retry")
            self.jlog(ev="submit_fail_confirmed_absent", oid=oid)
            self.st["open"] = None; self.save(); return
        log(f"submit failed but position is {self.pos_qty} — keeping state, flattening")
        await self.flatten("submit_fail_with_position")

    async def enter(self, kind, key):
        from async_rithmic import OrderType, TransactionType
        if self.st.get("halted"):
            log(f"{kind}: HALTED — no entries until manual clear"); return
        worst = WORST_STOPPED_PER_MICRO      # was 2100/1000 hardcoded, both stale
        if self.live and AUTHORISED_MAX.get(kind, 0) < 1:
            log(f"{kind}: not authorised for live trading (see AUTHORISED_MAX) — paper only")
            self.jlog(ev="unauthorised_leg", kind=kind, key=key)
            self.mark_done(kind, key); return     # settle it; do not retry all window
        # Refuse to OPEN a hold that would still be running when liquidity leaves the
        # contract. exit_date is the calendar day this window closes.
        try:
            exit_on = self._exit_date(kind, key)
        except (KeyError, ValueError) as e:
            log(f"{kind}: cannot determine exit date for {key!r} ({e}) — refusing entry")
            self.jlog(ev="bad_key_block", kind=kind, key=key, err=str(e))
            self.mark_done(kind, key); return
        if self.live and exit_on >= self.roll_date:
            log(f"{kind}: window ends on/after the {self.roll_date} roll — skipping")
            self.jlog(ev="roll_block", kind=kind, key=key, roll=str(self.roll_date))
            self.mark_done(kind, key); return
        if self.live:
            cushion = await self._broker_cushion()
            if cushion is None:
                log(f"{kind}: broker cushion UNAVAILABLE — entry blocked (fail-safe)")
                self.jlog(ev="cushion_unavailable_block", kind=kind); return
        else:
            cushion = cushion_for(self.st["banked"], self.st.get("peak_banked"))
        # Size from the cushion we actually verified. This used to size off the internal
        # estimate and only gate on the broker number, so the two could disagree -- and
        # the gate ran at 0.75 while the documented policy (TAIL_BUDGET) is 0.60.
        qty = ladder_size(kind, self.st["banked"], cushion=cushion,
                          peak_banked=self.st.get("peak_banked"), live=self.live)
        if qty < 1:
            log(f"{kind}: cushion ${cushion:.0f} too thin for one micro at "
                f"{TAIL_BUDGET:.0%} tail budget — skipping")
            self.jlog(ev="cushion_block", kind=kind, cushion=cushion); return
        if qty * worst > TAIL_BUDGET * cushion:      # belt-and-braces; ladder_size honours it
            log(f"{kind}: cushion gate blocked entry ({qty*worst} vs {TAIL_BUDGET*cushion:.0f})")
            self.jlog(ev="cushion_block", kind=kind, qty=qty, cushion=cushion); return
        blocked = self.consistency_block(kind)
        if blocked:
            log(f"{kind}: {blocked}"); self.jlog(ev="consistency_block", kind=kind, why=blocked)
            return
        ref, age = self.quote()
        if ref is None or not math.isfinite(ref) or age > 300:
            log(f"{kind}: no fresh two-sided quote (age {age:.0f}s) — retry next loop")
            self.jlog(ev="stale_quote_skip", kind=kind, age=age)
            push(f"{kind}: no fresh quote (age {age:.0f}s) — entry skipped")
            return
        oid = f"{kind.lower()}-{uuid.uuid4().hex[:6]}"
        self.st["open"] = {"kind": kind, "key": key, "oid": oid, "qty": qty, "ref": ref,
                           "fill": None, "verified": False, "stp_sent": False,
                           "contract": self.contract,
                           "entered_at": time.time(), "verify_after": time.time() + 5}
        self.save()
        if self.live:
            self.pos_qty = None; self.pos_qty_ts = 0.0   # invalidate BEFORE the order exists
            try:
                await self._call("submit_order", self.client.plants["order"].submit_order(
                    order_id=oid, symbol=self.contract, exchange=EXCHANGE, qty=qty,
                    transaction_type=TransactionType.BUY, order_type=OrderType.MARKET,
                    account_id=self.acct))
            except Exception as e:
                # DO NOT assume the order never reached the exchange. A timeout on the
                # response is indistinguishable from a rejection, and clearing state
                # here let the next 15s tick re-enter -- two positions, one tracked,
                # one stop. Reconcile against the broker before deciding.
                log(f"{kind} submit FAILED {e} — reconciling before any retry")
                self.jlog(ev="submit_fail", err=str(e), oid=oid)
                await self._reconcile_after_submit_failure(oid, kind, key)
                return
            log(f"LIVE ENTRY {kind} {qty} {self.contract} @~{ref}")
        else:
            self.st["open"]["fill"] = ref      # dry: simulate a full instant fill so the
            self.st["open"]["filled_qty"] = qty  # protection lifecycle is exercised end-to-end
            log(f"DRY ENTER {kind} {qty} @~{ref}")
        self.jlog(ev="entry", kind=kind, qty=qty, ref=ref, live=self.live)

    STALE_POSITION_SEC = 30 * 60

    def _stale(self, ot):
        """Is this position record too old to act on without re-checking the broker?

        A `state.json` carrying a position from weeks ago is not something to place
        orders against. protect() was NOT gated on `halted` -- only enter() was -- so an
        orphan with verified=False made the daemon submit a STOP_MARKET SELL priced off
        a three-week-old fill. If spot had since fallen past that trigger, the "stop"
        became an immediate uncovered short with nothing protecting it.
        """
        return time.time() - float(ot.get("entered_at", 0)) > self.STALE_POSITION_SEC

    async def protect(self, ot):
        from async_rithmic import OrderType, TransactionType, OrderDuration
        if self.st.get("halted"):
            log("protect: HALTED — refusing to place orders from halted state"); return
        if self._stale(ot):
            log("protect: position record is stale — refusing to price a stop off it")
            push("CRITICAL stale position record — daemon halted, CHECK ACCOUNT")
            self.jlog(ev="protect_refused_stale", age_s=time.time() - float(ot.get("entered_at", 0)))
            self.st["halted"] = True; self.save(); return
        stop_px = round((ot["fill"] - CAT_STOP_PTS)/TICK)*TICK
        ot["stop_px"] = stop_px
        if self.live:
            # A stop above the current market is not a stop -- it is a market short.
            # Cheap sanity check before sending; the price feed is already to hand.
            ref, age = self.quote()
            if ref is not None and math.isfinite(ref) and age <= 300 and stop_px >= ref:
                log(f"CRITICAL stop {stop_px} >= market {ref} — refusing, flattening instead")
                self.jlog(ev="protect_refused_inverted", stop=stop_px, ref=ref)
                await self.flatten("stop_would_invert"); return
            try:
                await self._call("submit_stop", self.client.plants["order"].submit_order(
                    order_id=f"{ot['oid']}-stp", symbol=ot.get("contract", self.contract),
                    exchange=EXCHANGE,
                    qty=ot.get("filled_qty", ot["qty"]), transaction_type=TransactionType.SELL,
                    order_type=OrderType.STOP_MARKET, trigger_price=stop_px,
                    # GTC, explicitly. submit_order defaults to duration=DAY; these holds
                    # run up to 22h across a CME trade-date rollover, so a DAY stop can
                    # expire mid-position. It happened to survive for both windows, but
                    # only by accident of where the session boundary falls.
                    duration=OrderDuration.GTC,
                    account_id=self.acct))
                ot["verify_after"] = time.time() + 4
                log(f"cat-stop placed @ {stop_px} (GTC)")
            except Exception as e:
                log(f"CRITICAL protect fail {e}"); push(f"CRITICAL {ot['kind']} protect fail")
                await self.flatten("protect_fail")
        else:
            ot["verified"] = True; ot["verify_after"] = time.time() + RECHECK_SEC
            log(f"DRY cat-stop @ {stop_px}")

    def _validate_stop(self, orders, ot):
        """A resting stop that matches tag AND side(SELL)/qty/symbol/trigger."""
        want_tag = f"{ot['oid']}-stp"
        need_qty = int(ot.get("filled_qty", ot["qty"]))
        want_px = ot.get("stop_px")
        for o in orders or []:
            if str(getattr(o, "user_tag", "") or "") != want_tag:
                continue
            status = str(getattr(o, "status", "") or "").lower()
            if any(k in status for k in ("cancel", "complete", "fill", "reject", "done", "inactive", "expire")):
                return (False, f"status={status}")
            if getattr(o, "symbol", self.contract) not in ("", self.contract):
                return (False, "symbol")
            tt = str(getattr(o, "transaction_type", "") or "")
            if tt and tt not in ("2", "SELL", "TransactionType.SELL"):
                return (False, f"side={tt}")
            q = getattr(o, "quantity", None)
            if q is None: q = getattr(o, "qty", None)
            if q is not None and int(q) != need_qty:
                return (False, f"qty={q}!={need_qty}")
            trg = getattr(o, "trigger_price", None) or getattr(o, "stop_price", None)
            if trg is not None and want_px is not None and abs(float(trg) - want_px) > 5 * TICK:
                return (False, f"trigger={trg}!={want_px}")
            return (True, "ok")
        return (False, "missing")

    async def verify(self, ot):
        """Confirm the stop is resting, then KEEP confirming for the life of the trade.

        This used to run once: on success it set verified=True and verify_after=None,
        which made the caller's guard permanently false. A stop cancelled mid-hold --
        by the broker, a risk action, or a reconnect -- was never noticed again, so the
        position could run unprotected for the remaining ~22 hours. Re-arm the timer
        instead of disarming it.
        """
        orders = await self._list_orders()
        if orders is None:
            # Unknown != invalid. Retry soon; only a definite answer flattens.
            ot["verify_after"] = time.time() + 6
            ot["verify_unknown"] = int(ot.get("verify_unknown", 0)) + 1
            if ot["verify_unknown"] * 6 > 300:
                log("CRITICAL cannot read orders for 5min — cannot confirm stop, flattening")
                push(f"CRITICAL {ot['kind']} stop unverifiable 5min — flattening")
                self.jlog(ev="protection_unverifiable")
                await self.flatten("protection_unverifiable")
            return
        ot["verify_unknown"] = 0
        ok, why = self._validate_stop(orders, ot)
        if ok:
            first = not ot.get("verified")
            ot["verified"] = True
            ot["verify_after"] = time.time() + RECHECK_SEC     # re-arm, do not disarm
            if first:
                log("protection verified ✓"); push(f"{ot['kind']} protected ✓")
                self.jlog(ev="protection_verified")
        else:
            log(f"CRITICAL stop not valid ({why}) — flatten"); push(f"CRITICAL {ot['kind']} stop {why} — flattening")
            self.jlog(ev="protection_invalid", why=why)
            await self.flatten("protection_not_resting")

    async def flatten(self, why):
        self.jlog(ev="flatten", why=why, live=self.live)
        ot = self.st.get("open")
        if ot: ot["closing"] = True; self.save()
        if self.live:
            confirmed = False
            t0 = time.time()                               # exit-request timestamp
            # Exit the contract the POSITION is in, which after a roll is not necessarily
            # the one we would trade today.
            sym = (ot or {}).get("contract") or self.contract
            for attempt in range(4):                       # verification sweep
                await self._cancel_all()
                try:
                    await self._call("exit_position",
                                     self.client.plants["order"].exit_position(
                                         account_id=self.acct, symbol=sym,
                                         exchange=EXCHANGE))
                except Exception:
                    pass                                   # _call already logged it
                await asyncio.sleep(4)
                orders = await self._list_orders()
                orders_clear = orders is not None and not self._our_working_tags(orders)
                # a FRESH (post-exit) zero only — a stale 0 from before entry never counts
                fresh = getattr(self, "pos_qty_ts", 0.0) >= t0
                pos_clear = (self.pos_qty == 0 and fresh)
                if orders_clear and pos_clear:
                    confirmed = True; break
                log(f"flatten sweep {attempt+1}: orders_clear={orders_clear} pos={self.pos_qty} fresh={fresh}")
            if not confirmed:
                log("CRITICAL: flatten UNCONFIRMED — HALTING (state preserved)")
                push("CRITICAL: flatten unconfirmed — CHECK ACCOUNT NOW (daemon halted)")
                self.jlog(ev="flatten_unconfirmed_halt")
                self.st["halted"] = True; self.save()
                return
            log(f"FLATTEN ({why}) confirmed={confirmed}")
            # Give the exit FILL a moment to arrive and be booked by _note(). Clearing
            # st["open"] first made _note() see ot=None and DROP the realised PnL --
            # `banked` then drifts, and `banked` is what size_cap() reads. Each sweep
            # only sleeps 4s, so under reconnect delay this was a routine loss of the
            # accounting, not an edge case.
            for _ in range(10):
                if self.st.get("open") is None:
                    return                                 # _note() booked it; done
                await asyncio.sleep(1)
            if self.st.get("open") is not None:
                # No fill notification arrived. Book at the best price we have rather
                # than silently losing the trade from the ledger, and say so.
                px, age = self.quote()
                mark = px if (px and math.isfinite(px) and age <= 300) else ot.get("fill") or ot["ref"]
                fqty = ot.get("filled_qty", ot["qty"])
                pnl = (float(mark) - (ot.get("fill") or ot["ref"]))*PV*fqty - 2*COMM*fqty
                log(f"FLATTEN booked from mark (no fill notification): pnl≈${pnl:+.2f}")
                self.jlog(ev="closed_from_mark", kind=ot["kind"], mark=mark, why=why)
                push(f"{ot['kind']} closed ~{pnl:+.2f} (marked, no fill msg)")
                self.book_pnl(pnl, ot["kind"], self.done_key(ot["kind"], ot["key"]), why)
            return
        if ot:
            px, _ = self.quote()
            fqty = ot.get("filled_qty", ot["qty"])
            pnl = ((px or ot["ref"]) - (ot.get("fill") or ot["ref"]))*PV*fqty - 2*COMM*fqty
            log(f"DRY FLATTEN {ot['kind']} pnl≈${pnl:+.2f} ({why})")
            self.book_pnl(pnl, ot["kind"], self.done_key(ot["kind"], ot["key"]), why)
        else:
            self.st["open"] = None; self.save()

    def schedule(self, now):
        """Return (kind, key, is_exit) for anything due at `now`.

        All ET, from a DST-aware clock (pd.Timestamp.now(tz=America/New_York)), so every
        boundary below is a true wall-clock ET minute in both halves of the year. The
        two-convention hazard in tzguard.py applies to the bar loaders, not here.
        """
        d, wd, hm = str(now.date()), now.weekday(), now.hour*60 + now.minute
        ot = self.st.get("open")
        # ---- exits first -------------------------------------------------------
        if ot:
            k = ot["kind"]
            # SAFETY NET, ahead of every scheduled exit: never let a position reach the
            # firm's mandatory 16:45 auto-liquidation. If a scheduled exit was missed
            # (daemon down, a key that no longer matches), this still closes the trade
            # at a price we chose. Nothing previously covered that case.
            # It stops AT the flatten, not at 18:00: between 16:45 and 18:00 the platform
            # accepts no orders, and the firm has already liquidated, so firing an exit
            # there just burns retries and ends in a false unconfirmed-flatten halt.
            if wd <= 4 and FLATTEN_HM - FLATTEN_MARGIN_MIN <= hm < FLATTEN_HM:
                return (k, ot["key"], True)
            # Each scheduled exit is bounded ABOVE by the flatten margin. An unbounded
            # `hm >= 960` meant a daemon restarting at 17:30 would fire an exit inside
            # the 16:45-18:00 closed window, where no orders are accepted.
            hard = FLATTEN_HM - FLATTEN_MARGIN_MIN
            if k == "WK" and wd == 0 and ot["key"] == d and 960 <= hm < hard:
                return ("WK", ot["key"], True)
            if k == "EO" and ot["key"] == d and 300 <= hm < hard:
                return ("EO", ot["key"], True)
            if k == "FOMC" and FOMC_EXIT.get(ot["key"]) == d and 835 <= hm < hard:
                return ("FOMC", ot["key"], True)
            return None
        # ---- entries (one position at a time, by design) ------------------------
        # 18:00, NOT 14:00. A 14:00 or 16:00 entry sits before the 16:45 ET mandatory
        # flatten and is force-closed the same evening -- the tested window can never
        # complete. tzguard.check_window_legal("18:00","13:55",1) is the check; 17:00 is
        # also illegal because the platform accepts no orders between 16:45 and 18:00.
        if wd == 6 and d not in NO_ENTRY_SUNDAYS:
            key = str((now + pd.Timedelta(days=1)).date())
            if not self.is_done("WK", key) and 1080 <= hm <= 1095:
                return ("WK", key, False)
        if d in FOMC_ENTRY and not self.is_done("FOMC", d) and 1080 <= hm <= 1090:
            return ("FOMC", d, False)
        if wd <= 4 and d not in HOLIDAYS and not self.is_done("EO", d) and 120 <= hm <= 130:
            if self._eo_condition(): return ("EO", d, False)
            self.mark_done("EO", d)
        return None

    def _eo_condition(self):
        try:
            fs = sorted(glob.glob(os.path.join(os.environ.get("RAW_DIR", ""), "stream_MNQ_*.csv")))
            rows = []
            for f in fs[-4:]:
                df = pd.read_csv(f, header=None, names=["ns","bp","bs","ap","asz"], usecols=[0,1,3])
                ts = pd.to_datetime(df["ns"], unit="ns", utc=True).dt.tz_convert(ET)
                mid = (df["bp"]+df["ap"])/2
                m = (ts.dt.hour*60+ts.dt.minute >= 570) & (ts.dt.hour*60+ts.dt.minute < 960)
                if m.sum() < 500: continue
                v = mid[m]
                rows.append({"day": f.split("_")[-1][:8], "ret": v.iloc[-1]-v.iloc[0], "rng": v.max()-v.min()})
            if not rows: return False
            dd = pd.DataFrame(rows).drop_duplicates("day").sort_values("day")
            last = dd.iloc[-1]
            return bool(last["ret"] < 0 and last["rng"] > dd["rng"].median())
        except Exception as e:
            log(f"eo condition err {e}"); return False

    async def startup_recovery(self):
        """Decide what a position record found in state.json on startup actually means.

        The old version flattened unconditionally. With no PNL feed yet, `pos_qty` is
        None, so the sweep could never confirm and the daemon halted with the record
        preserved -- then the fill-timeout branch in the loop re-entered flatten() every
        15 seconds forever, hammering exit_position, because a stale `entered_at` is
        always older than 90s. Ask the broker what is actually there first.
        """
        ot = self.st.get("open")
        if not ot:
            return
        age = time.time() - float(ot.get("entered_at", 0))
        log(f"startup with open state: {ot['kind']} {ot.get('qty')} age={age/3600:.1f}h")
        self.jlog(ev="startup_open_state", kind=ot["kind"], age_h=round(age/3600, 2))
        if not self.live:
            self.st["open"] = None; self.save(); return
        # Wait for a real PNL snapshot before judging anything.
        for _ in range(20):
            if self.pos_qty is not None:
                break
            await asyncio.sleep(1)
        if self.pos_qty is None:
            log("CRITICAL: no position feed at startup — cannot classify open state, HALTING")
            push("CRITICAL: stale open position, no position feed — daemon halted, CHECK ACCOUNT")
            self.jlog(ev="startup_no_feed_halt")
            self.st["halted"] = True; self.save(); return
        if self.pos_qty == 0:
            # Broker says flat. The record is an orphan from a crash; there is nothing
            # to flatten. Book it out of the ledger and clear, do not send orders.
            log("broker reports FLAT — clearing orphaned position record, no orders sent")
            push(f"orphaned {ot['kind']} record cleared (broker flat)")
            self.jlog(ev="startup_orphan_cleared", kind=ot["kind"], key=ot["key"])
            await self._cancel_all()
            self.mark_done(ot["kind"], ot["key"])
            self.st["open"] = None; self.save(); return
        log(f"broker reports {self.pos_qty} contracts — real position, flattening")
        push(f"recovered live position {self.pos_qty} — flattening")
        await self.flatten("startup_recovery")

    async def run(self):
        await self.connect()
        await self.startup_recovery()
        while True:
            self.beat = time.time()
            try:
                await asyncio.wait_for(self._tick(), timeout=WATCHDOG_SEC)
            except asyncio.TimeoutError:
                # The advertised watchdog never existed: self.beat was written and never
                # read, and no broker await had a timeout, so a hung call stalled the
                # loop silently with a position open.
                log(f"WATCHDOG: iteration exceeded {WATCHDOG_SEC}s")
                push("WATCHDOG: daemon iteration hung — check connectivity")
                self.jlog(ev="watchdog_timeout")
            if self._stop:
                return
            await asyncio.sleep(LOOP_SEC)

    async def _tick(self):
        now = self.now()
        if os.path.exists(self.kill_p):
            if self.st.get("open"): await self.flatten("kill")
            push("KILL — daemon down"); self._stop = True; return
        # A halted daemon must not keep sending orders. `halted` gated enter() only,
        # so the reconciliation paths below kept firing against dead state.
        if self.st.get("halted"):
            if not self._halt_logged:
                log("HALTED — no further action until state.json is cleared by hand")
                self._halt_logged = True
            return
        ot = self.st.get("open")
        # fill-timeout reconciliation: no fill at all, OR stuck partial fill
        if ot and self.live and time.time() - ot["entered_at"] > 90 \
           and ot.get("filled_qty", 0) == 0:
            log("fill timeout (no fill) — reconciling"); push("fill timeout — reconciling")
            await self.flatten("fill_timeout")
        ot = self.st.get("open")
        if ot and self.live and 0 < ot.get("filled_qty", 0) < ot.get("qty", 1) \
           and time.time() - ot["entered_at"] > 25:
            log(f"PARTIAL fill stuck {ot['filled_qty']}/{ot['qty']} — flatten")
            push("partial fill stuck — flattening")
            await self.flatten("partial_fill_timeout")
        # protection lifecycle: only once the order is FULLY filled. Runs for the LIFE
        # of the trade now -- `verified` no longer latches the check off.
        ot = self.st.get("open")
        if ot and self.live and ot.get("filled_qty", 0) >= ot.get("qty", 1) \
           and ot.get("verify_after") and time.time() >= ot["verify_after"]:
            if not ot.get("stp_sent"):
                await self.protect(ot)
                if self.st.get("open") is ot: ot["stp_sent"] = True
            else:
                await self.verify(ot)
            if self.st.get("open") is ot: self.save()
        due = self.schedule(now)
        if due:
            kind, key, is_exit = due
            if is_exit: await self.flatten(f"{kind}_scheduled_exit")
            else: await self.enter(kind, key)

def selftest():
    """Mock-clock walk through every branch in DRY mode."""
    import tempfile
    td = tempfile.mkdtemp()
    seq = [
        pd.Timestamp("2026-07-12 18:02", tz=ET),   # Sunday entry
        pd.Timestamp("2026-07-12 18:03", tz=ET),
        pd.Timestamp("2026-07-13 16:01", tz=ET),   # Monday exit
        pd.Timestamp("2026-07-14 02:03", tz=ET),   # Tue euro-open (condition mocked True)
        pd.Timestamp("2026-07-14 05:01", tz=ET),   # euro-open exit
        pd.Timestamp("2026-09-15 14:05", tz=ET),   # FOMC day, but 14:05 -- must NOT enter
        pd.Timestamp("2026-09-15 18:05", tz=ET),   # FOMC entry, 18:00 window
        pd.Timestamp("2026-09-16 13:56", tz=ET),   # FOMC exit
        pd.Timestamp("2026-09-06 18:05", tz=ET),   # holiday Sunday: must NOT enter
    ]
    i = [0]
    def now(): return seq[min(i[0], len(seq)-1)]
    d = Daemon(td, live=False, now_fn=now, quote_fn=lambda: (23000.0, 1.0))
    d._eo_condition = lambda: True
    async def drive():
        d.acct = "TEST"; d.contract = "MNQU6"
        for step in range(len(seq)):
            i[0] = step
            nowv = d.now()
            due = d.schedule(nowv)
            if due:
                kind, key, is_exit = due
                if is_exit: await d.flatten(f"{kind}_exit")
                else: await d.enter(kind, key)
        assert not d.st["open"], "position left open at end"
        assert "WK:2026-07-13" in d.st["done"], "weekend trade missing"
        assert "EO:2026-07-14" in d.st["done"], "euro-open missing"
        assert "FOMC:2026-09-15" in d.st["done"], "FOMC missing"
        assert "WK:2026-09-07" not in d.st["done"], "holiday entry fired!"
        # REGRESSION GUARD for the 14:00 entry bug. A 14:05 mark on an FOMC entry date
        # must NOT produce an entry: 14:00 sits before the 16:45 flatten and would be
        # force-closed the same evening. 17:30 must also be refused -- the platform
        # accepts no orders between 16:45 and 18:00. Only 18:00-18:10 may fire.
        probe = Daemon(tempfile.mkdtemp(), live=False,
                       now_fn=lambda: pd.Timestamp("2026-10-27 14:05", tz=ET),
                       quote_fn=lambda: (23000.0, 1.0))
        for t, want in (("14:05", None), ("16:50", None), ("17:30", None), ("18:05", "FOMC")):
            got = probe.schedule(pd.Timestamp(f"2026-10-27 {t}", tz=ET))
            kind = got[0] if got else None
            assert kind == want, f"FOMC entry at {t} ET: expected {want}, got {kind}"
        print("  regression guard: 14:05/16:50/17:30 refused, 18:05 fires — 18:00 entry locked in")
        # REGRESSION GUARD for the FOMC re-entry bug. schedule() tested `f"F{d}"` while
        # every completion path appended the bare `d`, so a completed or blocked FOMC
        # re-fired on the next 15s tick, over and over, inside the 18:00-18:10 window.
        # All three legs must go quiet once marked done, by the SAME key.
        t = pd.Timestamp("2026-10-27 18:05", tz=ET)
        probe2 = Daemon(tempfile.mkdtemp(), live=False, now_fn=lambda: t,
                        quote_fn=lambda: (23000.0, 1.0))
        assert probe2.schedule(t)[0] == "FOMC", "FOMC should fire before completion"
        probe2.mark_done("FOMC", "2026-10-27")
        assert probe2.schedule(t) is None, "FOMC RE-ENTERED after completion"
        for kind, when, key in (("WK", "2026-07-12 18:05", "2026-07-13"),
                                ("EO", "2026-07-14 02:05", "2026-07-14")):
            ts = pd.Timestamp(when, tz=ET)
            p = Daemon(tempfile.mkdtemp(), live=False, now_fn=lambda ts=ts: ts,
                       quote_fn=lambda: (23000.0, 1.0))
            p._eo_condition = lambda: True
            assert p.schedule(ts)[0] == kind
            p.mark_done(kind, key)
            assert p.schedule(ts) is None, f"{kind} RE-ENTERED after completion"
        print("  regression guard: WK/EO/FOMC all de-dup on one key — no re-entry loop")
        # 16:45 flatten safety net: an open position in the closed window must exit.
        p = Daemon(tempfile.mkdtemp(), live=False,
                   now_fn=lambda: pd.Timestamp("2026-07-13 16:41", tz=ET),
                   quote_fn=lambda: (23000.0, 1.0))
        p.st["open"] = {"kind": "WK", "key": "2026-07-13", "oid": "wk-x", "qty": 1,
                        "ref": 23000.0, "fill": 23000.0, "filled_qty": 1}
        got = p.schedule(pd.Timestamp("2026-07-13 16:41", tz=ET))
        assert got == ("WK", "2026-07-13", True), f"no 16:45 safety flatten, got {got}"
        print("  regression guard: 16:41 ET forces an exit ahead of the 16:45 flatten")
        print("SELFTEST PASS — all schedule branches exercised, holiday guard held,"
              f" banked=${d.st['banked']:+.2f}")
    asyncio.run(drive())

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--i-accept-unverified-rules", action="store_true",
                    help="proceed live even though account_rules.yaml is unconfirmed")
    a = ap.parse_args()
    if a.selftest: selftest(); return
    if a.live:
        assert_windows_legal()
        # account_rules.yaml drives the MLL, the flatten time, the sizing ladder and
        # every legality check in this file, and it is still web-sourced. It also states
        # `overnight_holding_permitted: false` -- and this entire book is overnight
        # holds. If that is meant literally, every trade here breaks a rule. Confirm it
        # against the dashboard before risking money.
        if not RULES_VERIFIED and not a.i_accept_unverified_rules:
            log("REFUSING --live: account_rules.yaml has "
                "verified_against_account_dashboard: false.")
            log("  Confirm the MLL, the 16:45 flatten, the consistency rule and whether "
                "overnight holds are permitted at all, then set that flag true.")
            log("  To override deliberately: --i-accept-unverified-rules")
            raise SystemExit(2)
        live_legs = {k: v for k, v in AUTHORISED_MAX.items() if v > 0}
        if not live_legs:
            log("REFUSING --live: no strategy is authorised (AUTHORISED_MAX is all 0).")
            log("  See the pre-registration notes above AUTHORISED_MAX in this file.")
            raise SystemExit(2)
        log(f"*** LIVE: ladder daemon, cat-stop {CAT_STOP_PTS}pts, "
            f"authorised {live_legs} ***")
        for k, v in AUTHORISED_MAX.items():
            if v == 0:
                log(f"    {k}: PAPER ONLY — not authorised by preregister.yaml")
    asyncio.run(Daemon(a.out_dir, a.live).run())

if __name__ == "__main__": main()
