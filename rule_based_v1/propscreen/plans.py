"""Prop-firm plan data. Every row carries its source and a confidence flag.

CONFIDENCE
  SOURCE   every field read off the firm's own site or help centre
  PARTIAL  geometry from the firm or a survey, but fee / split / breach unconfirmed
  ASSUMED  placeholder, present so the shape is visible -- DO NOT RANK ON IT

BREACH BASIS IS THE TRAP. Almost no firm states whether a breach is evaluated on
live equity or only on the closing balance, and at Lucid it took a direct answer
from the firm to settle. It is worth 0.68x vs 1.08x on pass probability and, worse,
0.29x vs 1.18x on extraction. So every unconfirmed row defaults to Breach.LIVE,
the conservative reading. A row that only looks good because it was assumed
close-only is exactly the mistake this whole screen exists to avoid.

SECOND TRAP, LEARNED 2026-09-13: a firm saying its floor "does not trail intraday"
is a statement about where the FLOOR SITS, not about what TESTS it. Lucid's floor
also only moved end-of-day, and it still breached on live equity. Do not read
"static drawdown" as "close-only breach". They are independent fields.

THIRD TRAP: payout friction is NOT uniform. Lucid wants 5 days of >=$200. Elite's
DTF 100k wants 20 qualifying days of >=$500 each, none below 50% of the best day,
against a $25,000 lifetime cap. The `extraction_efficiency` in model.py is
calibrated on LUCID'S friction and will overstate a firm with harsher rules.
"""
from __future__ import annotations

from datetime import date

from .model import Breach, Drawdown, Plan

LUCID = "https://support.lucidtrading.com/en/articles/12945815-lucidflex-drawdown"
ETF_DTF = "https://elitetraderfunding.app/help/how-the-dtf-plan-works"
ETF_STATIC = ("https://elitetraderfunding.app/blog/"
              "static-drawdown-prop-firms-what-it-means-and-which-offer-it-2026")
EPIC = "https://epicctrader.com/static-drawdown-prop-firms/"
PROPSCOPE = "https://propscope.net/en/prop-firms-no-trailing-drawdown/"
DAYTRADERS = "https://daytraders.trading/"
V = date(2026, 9, 13)

PLANS: list[Plan] = [
    # ================================================================== LUCID ==
    # The calibration rows. Every field confirmed at source, breach confirmed
    # directly by the firm. This is the only fully-known plan we have, and the
    # only one whose mechanics multipliers are MEASURED rather than extrapolated.
    Plan("Lucid", "Flex 100k", 100_000, 6_000, 3_000,
         Drawdown.TRAILING_EOD, Breach.LIVE, fee=225.0, fee_discounted=158.0,
         split=0.90, max_minis=6, payout_min_days=5, payout_min_daily=200.0,
         payout_cap=2_500.0, consistency=0.50, source=LUCID, verified=V,
         notes="CONFIDENCE=SOURCE. Breach on live equity confirmed by the firm; "
               "MLL location moves EOD. Measured P(pass) 22.8-25.8%, EV ~+$164 "
               "with 2020-26 beta and -$23 beta-stripped. OWNED. Do not re-buy."),
    Plan("Lucid", "Flex 150k", 150_000, 9_000, 4_500,
         Drawdown.TRAILING_EOD, Breach.LIVE, fee=315.0, fee_discounted=221.0,
         split=0.90, max_minis=10, payout_min_days=5, payout_min_daily=250.0,
         payout_cap=3_000.0, consistency=0.50, source=LUCID, verified=V,
         notes="CONFIDENCE=SOURCE. Same geometry ratio as the 100k, so same verdict."),

    # ====================================================== ELITE TRADER FUNDING ==
    # The find of this screen. The 100k DTF is the ONLY plan located that is
    # static AND assessed end-of-day rather than on every intraday tick.
    Plan("EliteTraderFund", "DTF 100k", 100_000, 5_100, 5_000,
         Drawdown.STATIC, Breach.CLOSE, fee=257.0, fee_discounted=77.10,
         split=1.00, activation_fee=None, max_minis=None,
         payout_min_days=20, payout_min_daily=500.0, payout_cap=25_000.0,
         consistency=0.50, source=ETF_DTF, verified=V,
         notes="CONFIDENCE=PARTIAL, and it is the top row so treat that seriously. "
               "CONFIRMED at the firm's help centre: no evaluation phase, $5,000 "
               "FIXED drawdown, 'the assessment occurs at end-of-day, not intraday', "
               "one-time fee with no monthly subscription and no activation fee, "
               "swing trading allowed with no flatten rule. The $5,100 'target' is "
               "the safety net (max DD + $100) you must reach in REALISED profit "
               "before any withdrawal, and it is not itself withdrawable. "
               "UNCONFIRMED: price. The firm's own page refuses to print one and "
               "epicctrader lists $997 against a promo listing of $257/$77.10 -- a "
               "10x disagreement. VERIFY BEFORE BUYING. "
               "UNMODELLED AND LIKELY DECISIVE: 20 qualifying days at >=$500 each, "
               "each >=50% of the best day, and a $25,000 lifetime cap across all "
               "accounts. That friction is far harsher than Lucid's 5 days at $200 "
               "which the efficiency term is calibrated on."),
    Plan("EliteTraderFund", "50k Static", 50_000, 4_000, 2_000,
         Drawdown.STATIC, Breach.CLOSE, fee=49.70, split=0.80, monthly=True,
         reset_fee=47.0, activation_fee=177.0, payout_min_days=5,
         source=ETF_STATIC, verified=V,
         notes="CONFIDENCE=PARTIAL. Firm states no daily loss limit, no consistency "
               "rule during evaluation, 5 minimum trading days, $47 reset, and that "
               "breach is measured 'against your closing balance rather than every "
               "intraday tick'. Billing is MONTHLY plus a one-time activation on "
               "passing; the $177 activation is read off a survey, not the firm."),
    Plan("EliteTraderFund", "25k Static", 25_000, 2_000, 1_000,
         Drawdown.STATIC, Breach.CLOSE, fee=27.70, split=0.80, monthly=True,
         reset_fee=47.0, activation_fee=177.0, payout_min_days=5,
         source=ETF_STATIC, verified=V, notes="CONFIDENCE=PARTIAL, as the 50k."),
    Plan("EliteTraderFund", "10k Static", 10_000, 1_000, 500,
         Drawdown.STATIC, Breach.CLOSE, fee=14.70, split=0.80, monthly=True,
         reset_fee=47.0, activation_fee=177.0, payout_min_days=5,
         source=ETF_STATIC, verified=V,
         notes="CONFIDENCE=PARTIAL. The $177 activation exceeds the entire "
               "extraction bound here, which is why it scores as it does."),

    # ================================================================ PHIDIAS ==
    # Static floor confirmed in the firm's own marketing. Breach basis NOT stated
    # anywhere, and phidiaspropfirm.com/rules returns 403 without a browser UA.
    Plan("Phidias", "E2L 25k", 25_000, 1_500, 500,
         Drawdown.STATIC, Breach.LIVE, fee=164.0, fee_discounted=55.40,
         split=0.80, max_minis=2, payout_min_days=0, source=PROPSCOPE, verified=V,
         notes="CONFIDENCE=PARTIAL. Static floor and zero minimum trading days "
               "stated by the firm; first payout converts to a LIVE funded account. "
               "Breach basis UNSTATED -- defaulted to LIVE. Rules page 403s without "
               "a browser user-agent, same as Lucid's help centre did."),
    Plan("Phidias", "E2L 50k", 50_000, 2_500, 650,
         Drawdown.STATIC, Breach.LIVE, fee=361.0, fee_discounted=144.60,
         split=0.80, max_minis=5, payout_min_days=0, source=EPIC, verified=V,
         notes="CONFIDENCE=PARTIAL. Breach basis UNSTATED."),
    Plan("Phidias", "E2L 150k", 150_000, 4_500, 1_000,
         Drawdown.STATIC, Breach.LIVE, fee=225.0, split=1.00, max_minis=9,
         payout_min_days=0, source=EPIC, verified=V,
         notes="CONFIDENCE=PARTIAL. Worst geometry of the three (D/T = 0.22)."),

    # ============================================================= DAYTRADERS ==
    # Cheapest one-time fees located, but the geometry is poor and there is a
    # $130 activation on top of every one of them.
    Plan("DayTraders", "25k Static", 25_000, 2_500, 750,
         Drawdown.STATIC, Breach.LIVE, fee=30.0, split=1.00, activation_fee=130.0,
         max_minis=4, source=DAYTRADERS, verified=V,
         notes="CONFIDENCE=PARTIAL. One-time fee, no subscription, plus a flat $130 "
               "activation to unlock the funded account. Breach basis UNSTATED."),
    Plan("DayTraders", "50k Static", 50_000, 3_750, 1_000,
         Drawdown.STATIC, Breach.LIVE, fee=40.0, split=1.00, activation_fee=130.0,
         max_minis=6, source=DAYTRADERS, verified=V, notes="CONFIDENCE=PARTIAL."),
    Plan("DayTraders", "100k Static", 100_000, 5_750, 1_500,
         Drawdown.STATIC, Breach.LIVE, fee=65.0, split=1.00, activation_fee=130.0,
         max_minis=8, source=DAYTRADERS, verified=V, notes="CONFIDENCE=PARTIAL."),
    Plan("DayTraders", "150k Static", 150_000, 6_750, 1_750,
         Drawdown.STATIC, Breach.LIVE, fee=80.0, split=1.00, activation_fee=130.0,
         max_minis=8, source=DAYTRADERS, verified=V, notes="CONFIDENCE=PARTIAL."),

    # ================================================================= BLUSKY ==
    Plan("BluSky", "Propel 150k", 150_000, 3_000, 1_000,
         Drawdown.STATIC, Breach.LIVE, fee=119.0, split=0.90, monthly=True,
         max_minis=1, source=EPIC, verified=V,
         notes="CONFIDENCE=PARTIAL. MONTHLY. Only 1 mini / 10 micros on a 150k, so "
               "reaching the target is slow, which is the worst pairing with monthly "
               "billing. BluSky's separate Orbit account is EOD-TRAILING once funded "
               "despite being listed among static plans -- do not confuse them."),
    Plan("BluSky", "Propel 200k", 200_000, 6_000, 2_000,
         Drawdown.STATIC, Breach.LIVE, fee=189.0, split=0.90, monthly=True,
         max_minis=2, source=EPIC, verified=V, notes="CONFIDENCE=PARTIAL. MONTHLY."),
    Plan("BluSky", "Propel 300k", 300_000, 20_000, 5_000,
         Drawdown.STATIC, Breach.LIVE, fee=224.0, split=0.90, monthly=True,
         max_minis=5, source=EPIC, verified=V,
         notes="CONFIDENCE=PARTIAL. MONTHLY. D/T = 0.25, the worst on the board."),

    # ======================================================= TRAILING CONTROLS ==
    # Included so the screen is never accused of only looking where it wants to.
    # All expected to score at or below Lucid.
    Plan("AlphaFutures", "Zero 50k", 50_000, 3_000, 2_000,
         Drawdown.TRAILING_EOD, Breach.LIVE, fee=139.0, fee_discounted=69.50,
         split=0.90, source=PROPSCOPE, verified=V,
         notes="CONFIDENCE=ASSUMED on target/drawdown. EOD MLL that locks at the "
               "start balance. Geometry NOT read at source -- do not rank on it."),
    Plan("E8Markets", "Zero Starter 50k", 50_000, 3_000, 2_000,
         Drawdown.TRAILING_EOD, Breach.LIVE, fee=178.0, fee_discounted=116.0,
         split=0.90, source=PROPSCOPE, verified=V,
         notes="CONFIDENCE=ASSUMED on target/drawdown. EOD dynamic, locks after a "
               "buffer. Geometry NOT read at source -- do not rank on it."),
    Plan("Bulenox", "Option 2 50k", 50_000, 3_000, 2_500,
         Drawdown.TRAILING_EOD, Breach.LIVE, fee=175.0, fee_discounted=19.25,
         split=0.90, monthly=True, source=PROPSCOPE, verified=V,
         notes="CONFIDENCE=ASSUMED on target/drawdown. Option 2 is EOD; OPTION 1 IS "
               "INTRADAY TRAILING, the worst mechanic on the board. Cheapest fee "
               "located ($19.25) which is the only reason it is worth a source read."),
]

# Still unscreened. Each needs: target, drawdown, drawdown TYPE, BREACH BASIS,
# fee (and whether monthly), split, and the payout friction.
TO_RESEARCH = [
    "MyFundedFutures (EOD drawdown; Starter vs Expert differ)",
    "Take Profit Trader (EOD on the test, INTRADAY once you pass -- verify)",
    "Tradeify (static reported, unconfirmed)",
    "TradeDay, Funded Futures Network, Legends Trading",
    "Apex, Topstep (trailing -- controls, expect REJECT/MARGINAL)",
]

# The three questions that settle any row. They cost one support ticket each and
# are worth more than any backtest -- Lucid's took a direct answer from the firm.
QUESTIONS = [
    "Is the drawdown truly STATIC, or does the floor move at any point?",
    "Is a breach evaluated on LIVE intraday equity, or only on the CLOSING balance?",
    "Is the fee ONE-TIME or MONTHLY, and is there an activation fee on passing?",
]
