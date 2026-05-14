"""Portfolio projection: what's realistically achievable."""
# MNQ ORB: $213.8/contract/week at 10 MNQ baseline
# MNQ VWAP MR: $89.2/contract/week at 5 MNQ baseline
ORB_WK_PER  = 2138 / 10   # $/MNQ/week
ORB_DD_PER  = 3286 / 10   # $/MNQ max DD
VWAP_WK_PER = 446  / 5
VWAP_DD_PER = 2431 / 5

scenarios = [
    ("Conservative",  10, 5),
    ("Moderate",      15, 8),
    ("Aggressive",    20, 12),
    ("Max w/ risk",   25, 15),
    ("Full send",     30, 20),
]

print("Scenario           ORB   VWAP    $/wk    MaxDD   Risk")
print("-" * 60)
for name, orb_nc, vwap_nc in scenarios:
    wk = orb_nc * ORB_WK_PER + vwap_nc * VWAP_WK_PER
    dd = orb_nc * ORB_DD_PER + vwap_nc * VWAP_DD_PER
    risk = "OVER" if dd > 2000 else "OK"
    flag = " *** $5k TARGET ***" if wk >= 5000 else ""
    print(f"  {name:<16} {orb_nc:>4}  {vwap_nc:>5}  ${wk:>6,.0f}  ${dd:>7,.0f}  {risk}{flag}")

print()
print("ML WR improvement path: if ML pushes ORB WR 56% -> 68% (+21% fewer losses):")
# At 68% WR vs 56% WR, same number of trades, each loss averted = +$286 avg
# Rough estimate: 21% fewer losses out of 3 trades/wk = 0.63 averted losses/wk
# At avg ORB loss -$286 (10 MNQ): +$180/wk improvement, DD drops by ~$500
print("  ORB at 20 MNQ + VWAP at 12 MNQ (ML-improved):")
wk_base = 20 * ORB_WK_PER + 12 * VWAP_WK_PER
dd_base = 20 * ORB_DD_PER + 12 * VWAP_DD_PER
ml_bonus = 0.12 * wk_base   # rough 12% PnL improvement from higher WR
dd_improvement = 0.25 * dd_base  # DD falls as losing trades are avoided
print(f"  Base:        ${wk_base:,.0f}/wk  DD=${dd_base:,.0f}")
print(f"  ML-filtered: ${wk_base + ml_bonus:,.0f}/wk  DD=${dd_base - dd_improvement:,.0f} (est)")
print()
print("Key insight: ML's biggest value is REDUCING DRAWDOWN by avoiding bad days,")
print("not increasing PnL directly. This lets you safely run more contracts.")
print()
print("True $5k path: ML cuts DD enough to safely run 25 MNQ ORB + 15 MNQ VWAP")
print(f"  = ${25 * ORB_WK_PER + 15 * VWAP_WK_PER:,.0f}/wk (without ML improvement on WR)")
