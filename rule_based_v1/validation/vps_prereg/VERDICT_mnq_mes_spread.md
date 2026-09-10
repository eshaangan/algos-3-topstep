# VERDICT — MNQ/MES market-neutral spread as a vehicle

**Status: NO-GO (2026-09-09). Killed on cost arithmetic, before any signal work.**
Script `rule_based_v1/validation/research_spread_cost_ratio.py`,
results `runs/spread_cost_ratio.json`.

Self-contained. This closes the *spread/hedged-execution* branch of the ledger's
continuation-queue item (2). It does **not** close cross-asset *conditioning*
(using ES state to gate an outright MNQ trade), which costs nothing extra and
remains open.

## The idea and why it was worth checking

Track B found the binding constraint is single-event tail risk against a $2-3k
trailing drawdown, not a shortage of alpha: the book caps at 1 micro, giving
~$132-232/wk against the $375-500/wk needed. A beta-neutral spread strips the
shared index move, cutting adverse excursion per position, which would permit
size. The obvious way for that to be wrong is that the spread has two legs and
costs more, so it must cut volatility FASTER than it raises cost.

## Result: it does not, and the penalty compounds

RTH 1-min bars, 2020-01 → 2025-12 (592,296 aligned). Hedge fit on dev (≤2023),
measured on val. Correlation 0.9166, variance-minimising **h = 1.43 MES per MNQ**.

| horizon | outright \|move\| | ratio | spread \|move\| | ratio | spread is |
|---|---|---|---|---|---|
| 5m | $31.97 | 6.4% | $12.24 | 45.9% | 7.2× worse |
| 30m | $85.68 | 2.4% | $31.78 | 17.7% | 7.4× worse |
| 60m | $125.87 | 1.6% | $46.95 | 12.0% | 7.5× worse |
| 240m | $332.46 | 0.6% | $138.86 | 4.0% | 6.5× worse |

Costs: outright $2.06/contract; package $5.61 = **2.72× the outright**. The vol
cut is only **2.4-2.7×**. The two effects MULTIPLY in a cost ratio, so the spread
runs ~6.5-7.5× worse at every horizon. The integer-hedge version (h=1, $4.55) is
worse still: a 1.85× vol cut against a 2.21× cost rise.

**Risk-matched, it is worse by more.** Sizing the spread up 2.5× to match the
outright's adverse excursion means paying 2.5 × 2.72 ≈ **6.8× the cost for the
same risk exposure**.

## The generalisable reason — this is why the branch closes

Hedging removes risk and return in the same proportion, but **commission is
charged per contract, not per unit of risk**. At $0.62/side the fixed per-lot
cost dominates, so any hedge that halves volatility more than doubles the cost of
the risk you keep. Market beta is not uncompensated risk you can discard for
free; it is the thing that pays.

⇒ **Do not build market-neutral or hedged structures on micro futures at this
commission.** The conclusion follows from the cost structure, not from this
particular pair, and would only reverse near ~$0.25/side.

## 🚨 A DATA BUG FOUND WHILE VERIFYING — read this before combining the tapes

**`mnq_1m_all.parquet` and `es_1min_eth_frontmonth.parquet` label bars
differently, and it is invisible in either series alone.**

Lag scan on 2025-06: the two agree at a **+59 minute** shift of the raw MNQ
stamp, **corr 0.9411**, and at **0.006** with the tz conversion alone.

    +60 min = Chicago -> Eastern
     -1 min = MNQ bars are stamped at bar CLOSE, ES bars at bar OPEN

Uncorrected, the measured 1-minute correlation is **0.014** — daily mean 0.007
across 1,535 sessions. The first cut of this study used that and concluded the
spread was dead for the wrong reason.

**The trap that nearly hid it:** the naive whole-sample correlation reads
**0.5596**, which looks plausible. That number comes entirely from the ~0.3% of
rows that straddle a session boundary, where both legs jump together. Always
compute cross-tape correlation on **adjacent rows within a session**.

Single-instrument results are unaffected — each tape is internally consistent,
and `research_fed_events.py` runs one instrument at a time. Any FUTURE combined
MNQ+ES work must apply the −1 minute shift.
