# VERDICT — stop x size grid on the repaired tape: **no configuration is worth deploying**

    date:    2026-09-10
    script:  rule_based_v1/validation/research_disaster_stop_v2.py
             (rewrite of the lost research_disaster_stop.py)
    report:  rule_based_v1/validation/disaster_stop_grid_report.py
    run:     runs/disaster_stop_repaired.json  (280 cells, 20,000 paths each)
    tape:    repaired mnq_1m_all.parquet (winter Sunday reopens restored)

## What was rebuilt

`research_disaster_stop.py` produced the frozen spec's 73.4% and then vanished
from the repo. It was never in git and survived only as bytecode, so its numbers
were unreproducible. This rewrite preserves the four things the original got
right, all of which matter:

* punitive gap fill at `min(stop, bar OPEN)`, so weekend gaps are never assumed
  tradeable;
* the path is truncated at the stop bar, so the account is not charged for an
  excursion a stopped position was never in;
* the EOD-trailing floor LOCKS at the starting balance,
  `floor = max(floor, min(peak - MLL, 0))`;
* intra-trade bust: `equity + MAE` is tested against the floor before the trade
  books, because the limit is breached live and not at settlement.

Legs are built once per stop level at one micro and scaled, since the stop is
per micro and both P/L and excursion are linear in size.

## Result — 0 of 280 cells clear 60%

| | p_pass >= 60% | >= 70% | >= 85% |
|---|---|---|---|
| cells out of 280 | **0** | **0** | **0** |

### The frozen spec's own cell ($600 stop, weekend x2, FOMC x2)

| horizon | claimed (damaged tape) | measured (repaired) |
|---|---:|---:|
| 16 weeks | 50.4% | **22.8%** |
| 32 weeks | 73.4% | **50.0%** |

### Best cell at each horizon

| horizon | config | p_pass | p_bust | decay50 | worst weekend |
|---|---|---:|---:|---:|---:|
| 16wk | $300 stop, wk x4, fomc x4 | 44.2% | **49.8%** | 32.6% | −$1,208 |
| 32wk | $500 stop, wk x2, fomc x4 | **57.5%** | 34.1% | 41.8% | −$1,004 |

**Searching the entire grid buys 7.5 points over the frozen config at 32 weeks
and nothing that changes the decision.** The best 16-week cell busts the account
half the time, which is not a strategy, it is a coin flip with extra steps.

## 🔧 The one actionable change: the $600 stop is dominated, use $300

Stop sweep at the 32-week optimum size (weekend x2, FOMC x4):

| stop | p_pass | p_bust | worst weekend | $/week |
|---|---:|---:|---:|---:|
| none | 55.0% | 41.8% | −$3,402 | 264.87 |
| **$300** | **56.4%** | **29.4%** | **−$604** | 211.67 |
| $400 | 56.0% | 33.4% | −$804 | 214.79 |
| $500 | 57.5% | 34.1% | −$1,004 | 229.73 |
| $600 | 53.7% | 39.4% | −$1,763 | 217.39 |
| $1000 | 54.1% | 41.3% | −$2,092 | 235.34 |

**$300 with FOMC x4 dominates the frozen $600 / x2 / x2 on every axis** — higher
pass probability (56.4% vs 50.0%), far lower bust (29.4% vs 38.9%), a worst
weekend of −$604 instead of −$1,763 (20% of the MLL instead of 59%), and a
better decay-stressed figure (40.0% vs 34.6%).

The original chose $600 on a tape whose worst weekend was missing. With the true
tail restored, the tighter stop is correct. Unstopped is the worst of all: it
buys $53/week of mean and pays for it with a −$3,402 weekend, which is 113% of
the drawdown limit, i.e. a single-weekend account kill.

## Bottom line

Nothing here clears 60%, at any stop, any size, either horizon. If the book is
traded at all, trade **$300 stop, weekend x2, FOMC x4, over 32 weeks**, and
understand that it is a 56% shot with a 29% chance of busting out. That is a
configuration choice, not an edge. The 85%-in-16-weeks target remains
unreachable and this closes the last route to it that had not been re-measured.
