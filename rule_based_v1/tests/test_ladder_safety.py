"""Live-safety regression tests for the ladder daemon (audit round 3 blockers)."""
import asyncio, tempfile, types
import pandas as pd
import pytest
from rule_based_v1.live import ladder_live_runner as L


def _daemon(live=True):
    d = L.Daemon(tempfile.mkdtemp(), live=live,
                 now_fn=lambda: pd.Timestamp("2026-07-13 16:00", tz=L.ET),
                 quote_fn=lambda: (23000.0, 1.0))
    d.acct = "ACCT-A"; d.contract = "MNQU6"; d.pos_qty = None
    return d


def _order(tag, sym="MNQU6", side="2", qty=1, trg=22550.0, status="working"):
    return types.SimpleNamespace(user_tag=tag, symbol=sym, transaction_type=side,
                                 quantity=qty, trigger_price=trg, status=status)


def test_flatten_halts_when_position_unknown():
    """B1: pos_qty=None must NOT be treated as flat — daemon halts, state kept."""
    d = _daemon()
    d.st["open"] = {"kind": "WK", "key": "k", "oid": "wk-1", "qty": 1, "ref": 23000.0,
                    "fill": 23000.0, "filled_qty": 1}
    d._cancel_all = lambda: asyncio.sleep(0)
    d.client = types.SimpleNamespace(plants={"order": types.SimpleNamespace(
        exit_position=lambda **k: asyncio.sleep(0))})
    d._list_orders = lambda: _aret([])          # orders clear
    d.pos_qty = None                            # position NEVER confirmed
    asyncio.run(d.flatten("test"))
    assert d.st.get("halted") is True
    assert d.st["open"] is not None             # state preserved, not cleared


def test_flatten_confirms_only_on_zero_position():
    d = _daemon()
    d.st["open"] = {"kind": "WK", "key": "k", "oid": "wk-1", "qty": 1, "ref": 23000.0, "fill": 23000.0, "filled_qty": 1}
    d._cancel_all = lambda: asyncio.sleep(0)
    d.client = types.SimpleNamespace(plants={"order": types.SimpleNamespace(exit_position=lambda **k: asyncio.sleep(0))})
    d._list_orders = lambda: _aret([])
    d.pos_qty = 0
    asyncio.run(d.flatten("test"))
    assert not d.st.get("halted")
    assert d.st["open"] is None


def test_pnl_ignores_other_account():
    """B3: a foreign account's MNQU6 position must not set pos_qty."""
    d = _daemon()
    n = types.SimpleNamespace(account_id="OTHER", symbol="MNQU6", open_position_quantity=7)
    asyncio.run(d._pnl_note(n))
    assert d.pos_qty is None
    n2 = types.SimpleNamespace(account_id="ACCT-A", symbol="MNQU6", open_position_quantity=2)
    asyncio.run(d._pnl_note(n2))
    assert d.pos_qty == 2


def test_stop_validation_rejects_wrong_qty():
    """B5: a resting order matching only the tag but wrong qty is NOT valid."""
    d = _daemon()
    ot = {"oid": "wk-1", "qty": 2, "filled_qty": 2, "stop_px": 22550.0}
    assert d._validate_stop([_order("wk-1-stp", qty=1)], ot)[0] is False   # 1 != 2
    assert d._validate_stop([_order("wk-1-stp", qty=2)], ot)[0] is True
    assert d._validate_stop([_order("wk-1-stp", side="1", qty=2)], ot)[0] is False  # BUY side
    assert d._validate_stop([_order("wk-1-stp", qty=2, trg=99999)], ot)[0] is False  # bad trigger
    assert d._validate_stop([], ot)[0] is False                            # missing


def test_broker_cushion_selects_pinned_account():
    """B4: cushion must read the pinned account's record, not a foreign one."""
    d = _daemon()
    d.acct_balance = 50000.0
    recs = [types.SimpleNamespace(account_id="OTHER", min_account_balance=99000.0),
            types.SimpleNamespace(account_id="ACCT-A", min_account_balance=47000.0)]
    d.client = types.SimpleNamespace(plants={"order": types.SimpleNamespace(
        get_account_rms=lambda: _aret(recs))})
    c = asyncio.run(d._broker_cushion())
    assert c == pytest.approx(3000.0)          # 50000 - 47000 (ACCT-A), not OTHER


def test_broker_cushion_none_when_balance_unknown():
    d = _daemon()
    recs = [types.SimpleNamespace(account_id="ACCT-A", min_account_balance=47000.0)]
    d.client = types.SimpleNamespace(plants={"order": types.SimpleNamespace(
        get_account_rms=lambda: _aret(recs))})
    assert asyncio.run(d._broker_cushion()) is None    # no acct_balance -> fail-safe


async def _aret_coro(v): return v
def _aret(v): return _aret_coro(v)
