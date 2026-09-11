"""Seed plan data. Every row carries its source and a confidence flag.

CONFIDENCE
  SOURCE   every field read off the firm's own site or help centre
  PARTIAL  geometry from the firm, but fee / split / breach basis not confirmed
  ASSUMED  placeholder, present so the shape is visible -- DO NOT RANK ON IT

BREACH BASIS IS THE TRAP. Almost no firm states whether a breach is evaluated on
live equity or only on the closing balance, and at Lucid it took a direct answer
from the firm to settle. It is worth 0.68x vs 1.08x on pass probability and, worse,
0.29x vs 1.18x on extraction. So every unconfirmed row defaults to Breach.LIVE,
the conservative reading. A row that only looks good because it was assumed
close-only is exactly the mistake this whole screen exists to avoid.
"""
from __future__ import annotations

from datetime import date

from .model import Breach, Drawdown, Plan

LUCID = "https://support.lucidtrading.com/en/articles/12945815-lucidflex-drawdown"
ETF = ("https://elitetraderfunding.app/blog/"
       "static-drawdown-prop-firms-what-it-means-and-which-offer-it-2026")

PLANS: list[Plan] = [
    # --- the calibration row: every field confirmed at source, breach confirmed
    #     directly by the firm. This is the only fully-known plan we have.
    Plan("Lucid", "Flex 100k", 100_000, 6_000, 3_000,
         Drawdown.TRAILING_EOD, Breach.LIVE, fee=225.0, fee_discounted=158.0,
         split=0.90, max_minis=6, payout_min_days=5, payout_min_daily=200.0,
         payout_cap=2_500.0, consistency=0.50, source=LUCID,
         verified=date(2026, 9, 10),
         notes="CONFIDENCE=SOURCE. Breach on live equity confirmed by the firm; "
               "MLL location moves EOD. Measured P(pass) 22.8%, EV +$20."),
    Plan("Lucid", "Flex 150k", 150_000, 9_000, 4_500,
         Drawdown.TRAILING_EOD, Breach.LIVE, fee=315.0, fee_discounted=221.0,
         split=0.90, max_minis=10, payout_min_days=5, payout_min_daily=250.0,
         payout_cap=3_000.0, consistency=0.50, source=LUCID,
         verified=date(2026, 9, 10), notes="CONFIDENCE=SOURCE."),

    # --- static-drawdown candidates. Geometry is from the firm; fee and split are
    #     NOT confirmed and ETF bills MONTHLY rather than once, which this model
    #     does not yet represent.
    Plan("EliteTraderFund", "50k Static", 50_000, 4_000, 2_000,
         Drawdown.STATIC, Breach.LIVE, fee=150.0, split=0.80, max_minis=4,
         reset_fee=47.0, source=ETF, verified=date(2026, 9, 10),
         notes="CONFIDENCE=PARTIAL. Geometry from the firm. Fee is a PLACEHOLDER "
               "and billing is MONTHLY, not one-time -- the model prices a single "
               "fee and will overstate EV for a slow campaign."),
    Plan("EliteTraderFund", "25k Static", 25_000, 2_000, 1_000,
         Drawdown.STATIC, Breach.LIVE, fee=100.0, split=0.80, max_minis=2,
         reset_fee=47.0, source=ETF, verified=date(2026, 9, 10),
         notes="CONFIDENCE=PARTIAL, as above."),
    Plan("EliteTraderFund", "10k Static", 10_000, 1_000, 500,
         Drawdown.STATIC, Breach.LIVE, fee=50.0, split=0.80, max_minis=1,
         reset_fee=47.0, source=ETF, verified=date(2026, 9, 10),
         notes="CONFIDENCE=PARTIAL, as above."),
]

# Firms named in the 2026 no-trailing / static surveys that still need a rulebook
# read. Each needs: target, drawdown, drawdown type, BREACH BASIS, fee, split.
TO_RESEARCH = [
    "Bulenox (Option 2 is EOD; Option 1 is intraday trailing)",
    "Alpha Futures (EOD trailing)",
    "MyFundedFutures (EOD drawdown; Starter vs Expert differ)",
    "Phidias (static / EOD options)",
    "E8 Markets (EOD dynamic)",
    "Take Profit Trader (EOD)",
    "Tradeify (static reported, unconfirmed)",
    "TradeDay, Blusky, Funded Futures Network, Legends Trading",
    "Apex, Topstep (trailing -- include as controls, expect REJECT/MARGINAL)",
]
