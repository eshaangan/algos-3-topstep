"""Kelly-based position sizing utilities."""

from __future__ import annotations

import math


def kelly_fraction(win_probability: float, payoff_ratio: float) -> float:
    """Return full Kelly fraction f = (p*b - q) / b."""
    p = float(win_probability)
    b = float(payoff_ratio)
    if b <= 0:
        return 0.0
    q = 1.0 - p
    return (p * b - q) / b


def fractional_kelly_contracts(
    bankroll_usd: float,
    risk_per_contract_usd: float,
    win_probability: float,
    payoff_ratio: float,
    fraction: float = 0.25,
    max_fraction_of_bankroll: float = 0.20,
    min_contracts: int = 1,
    max_contracts: int | None = None,
) -> dict:
    """Compute contract size from fractional Kelly with practical caps."""
    if bankroll_usd <= 0 or risk_per_contract_usd <= 0:
        return {"contracts": min_contracts, "reason": "invalid_inputs", "fractional_kelly": 0.0}

    raw_kelly = kelly_fraction(win_probability, payoff_ratio)
    frac = max(0.0, raw_kelly * float(fraction))
    frac = min(frac, float(max_fraction_of_bankroll))

    dollars_at_risk = bankroll_usd * frac
    contracts = int(math.floor(dollars_at_risk / risk_per_contract_usd))
    contracts = max(int(min_contracts), contracts)
    if max_contracts is not None:
        contracts = min(contracts, int(max_contracts))

    return {
        "contracts": int(contracts),
        "raw_kelly": float(raw_kelly),
        "fractional_kelly": float(frac),
        "reason": "kelly" if raw_kelly > 0 else "negative_edge_floor",
    }
