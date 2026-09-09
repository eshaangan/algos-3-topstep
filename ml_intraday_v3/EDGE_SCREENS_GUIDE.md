# Edge Attribution Screens

Three retrospective screens that run over trade logs already on disk. They ask
questions the existing PBO / DSR / CPCV gauntlet does not, and they need no new
market data beyond the bar tape the original backtest used.

Sourced from the Quant Guild library (see
`Quant_Guild_Library_to_V3_Report.md` for the earlier, unexecuted 2025 pass):

| Screen | Question | Origin |
|---|---|---|
| `analysis/alpha_beta.py` | Is the P/L just directional exposure, or exposure to realized range? | lecture 96, "I Bet You've Never Found Alpha" |
| `analysis/stability.py` | Is the P/L distribution the same at the end of the sample as at the start? | lecture 77, "Profitable vs Tradable" |
| `analysis/edge_stress.py` | Does the t-stat survive dropping any one period, and how concentrated is the P/L? | lectures 77 + 96 |

These are **complements, not replacements**. DSR/PBO ask "is this Sharpe real
given how many things I tried?". These ask "what is the P/L actually made of?"
and "did the generating process move?". A strategy can pass one and fail the other.

---

## 1. Alpha/beta attribution

Regresses daily strategy P/L on the instrument's own intraday move:

```
Spec 1 (directional):   pnl_t = alpha + beta * mkt_t + e_t
Spec 2 (+ convexity):   pnl_t = alpha + beta * mkt_t + gamma * |mkt_t| + e_t
```

Spec 2 is the addition that matters most here. A breakout strategy (ORB and
relatives) is structurally long realized range: it gets paid whenever the tape
moves, in either direction. Without the `|mkt_t|` term that volatility
harvesting is misattributed to alpha. Alpha that survives **both** specs is
return unexplained by direction *or* by range.

Design decisions worth knowing:

- **Newey-West (HAC) standard errors.** Daily strategy P/L is serially
  correlated (regime clustering, position carryover). Plain OLS errors would
  overstate t-stats, which is precisely the failure this screen exists to catch.
  Lag defaults to the Newey-West (1994) rule, `floor(4*(T/100)^(2/9))`.
- **The market move excludes the overnight gap.** It is measured within the
  session, because intraday strategies are flat overnight and never held that
  exposure.
- **Flat days are zero-filled by default.** A day the strategy sat out is still
  an observation on its equity curve. Pass `--traded-days-only` to restrict to
  traded days; the total P/L is unchanged, only the day count and mean move.
- **`beta_contracts_equiv`** divides beta by the point value, so beta reads as
  an average net contract count. Only meaningful when `--point-value` is the
  contract multiplier (MES 5.0, MNQ 2.0), not the strategy's aggregate sizing.

**Verdicts:** `ALPHA_SURVIVES`, `CONVEXITY_NOT_ALPHA` (paid for range, not
skill), `BETA_ONLY` (directional exposure), `NO_SIGNAL`.

## 2. P/L distribution stability

Splits the trade log into K chronological equal-count blocks, bins P/L on pooled
quantile edges, and computes Jensen-Shannon divergence between block histograms.

- **JS rather than raw KL** for the headline: symmetric and bounded in
  `[0, ln 2]`, so it is comparable across configs with different trade counts
  and P/L scales. Raw KL is available in the module.
- **Permutation null is the essential part.** Shuffle trade order, recompute,
  repeat. Two 15-trade blocks differ substantially by chance alone, so without
  this every small-n strategy looks unstable. The reported p-value is the share
  of shuffles whose divergence meets or exceeds the observed one.
- **Law of Total Expectation table** per block localises the drift:

  ```
  E[P/L] = E[P/L | win] * P(win) + E[P/L | loss] * P(loss)
  ```

  which tells you whether a strategy decayed because its hit rate collapsed,
  its winners shrank, or its losers grew — three different diagnoses.

**Verdicts:** `STABLE`, `UNSTABLE`, `UNKNOWN` (null skipped).

---

## How to run

```bash
# Both screens, trade log + benchmark tape
python -m ml_intraday_v3.diagnostics.run_edge_screens \
    --trades ml_intraday_v3/results/informed_flow_ml_trades.parquet \
    --pnl-col pnl_dollars \
    --bars data/processed/mes_2026_ytd_5m.h5 --bars-key bars_5min \
    --instrument MES \
    --out ml_intraday_v3/results/edge_screens/informed_flow.json

# Stability only (trade log has no usable timestamps)
python -m ml_intraday_v3.diagnostics.run_edge_screens \
    --trades ml_intraday_v3/diagnostics/orb_oos_results.json \
    --json-key trades --pnl-col pnl --no-time --blocks 2 --bins 6 \
    --out ml_intraday_v3/results/edge_screens/orb_oos.json
```

Notebook: section **4.7.2** of `ml_intraday_v3_pipeline_runner_enhanced.ipynb`.

Tests (41): `python -m pytest ml_intraday_v3/tests/test_alpha_beta.py ml_intraday_v3/tests/test_stability.py ml_intraday_v3/tests/test_edge_stress.py -q`

---

## Baseline results (2026-09-08)

Run over trade logs already in the repo. Artifacts in
`ml_intraday_v3/results/edge_screens/`.

### `informed_flow_ml_trades.parquet` — BETA_ONLY, the clearest case

65 trades, **100% long**, Feb–May 2026, total P/L **+$113**.

```
Spec 1:  alpha  -0.35  (p=0.97)     beta  0.777  t=3.34  p=0.0015 *   R^2=0.275
Spec 2:  alpha -35.68  (p=0.012)*   beta  0.650  t=3.57  p=0.0008 *
         gamma_abs 0.736  t=3.18  p=0.0024 *                          R^2=0.392
P/L attributed to market move: $134   P/L attributed to alpha: -$21
```

The entire P/L is the market move. Alpha is indistinguishable from zero in
Spec 1 and turns **significantly negative** in Spec 2 — once both direction and
range are controlled, the strategy loses money net of costs. Stability: STABLE
(p=0.33), i.e. it is consistently null rather than decayed. Two different
diseases, and this screen separates them.

### `ml_scalper_v3_oos_trades.parquet` — UNSTABLE

54 trades, 100% long, Mar–May 2026, +$24,483 headline.

```
mean pairwise JS = 0.1478   (null mean 0.0824, null q95 0.1255)   p = 0.0087
block  n   mean_pnl  P(win)   avg_win   avg_loss
  0   18    186.34   0.444   1544.24   -899.98
  1   18    973.43   0.722   1536.28   -489.99
  2   18    200.41   0.444   1238.90   -630.39
```

The middle block carries $17.5k of the $24.5k; blocks 0 and 2 average ~$190/trade.
This independently reproduces the Jun 2026 edge-decay verdict
(`project_ml_scalper_v3_edge_decay_jun2026`) **from the trade log alone**, with no
retrain and no replay. Alpha/beta was inconclusive here — the available MES tape
ends 2026-04-10 while the trades run to 05-11, so the benchmark is truncated.

### `analysis/v3_optimized_backtest_trades.csv` — headline P/L is not significant

493 trades, May 2019 – Apr 2020, **+$142,771** headline.

```
Spec 1:  alpha 590.05  se=384.34  t=1.54  p=0.126   (not significant)
mean pairwise JS p = 0.0447  -> UNSTABLE
avg winner: 949.7 -> 251.0 from first block to last
```

$142k of backtest profit that does not survive HAC inference on daily P/L,
because it is carried by a handful of huge days in the Feb–Mar 2020 vol event.
Beta is ~0, so this is not market exposure — it is a small number of outliers,
which is also a direct Topstep consistency-rule problem.

### `orb_oos_results.json` — STABLE, but underpowered

32 trades, blocks of 16. Mean pairwise JS 0.126 vs null q95 0.134, p = 0.066 —
does not clear the bar, but sits right at it. The block table shows hit rate
moving 0.375 -> 0.750 and average loser -106.9 -> -207.5. At n=32 this screen
cannot separate that from noise; treat it as "not enough evidence", not as a
clean bill of health.

---

## Limits

- The screens are **retrospective diagnostics on realized trades**. None
  produces a signal, and neither can create edge where there is none.
- Alpha/beta needs >= 10 aligned days and a bar tape covering the trade range.
- Stability needs >= 5 trades per block; below ~30 trades total it has low power
  and says so in its notes.
- A `STABLE` verdict on a null strategy means "consistently null". Always read
  it alongside the alpha/beta verdict, not on its own.


---

## 3. Edge robustness stress (`analysis/edge_stress.py`)

Added after the first two screens proved insufficient on the one strategy in this
repo that still had a live claim on being real.

- **Leave-one-period-out**: recompute mean and t with each calendar period
  dropped. Judge `worst_loo_t`, not the full-sample t.
- **Concentration**: top-k share of total P/L and a Gini over the winners. Note
  a top-k share can exceed 100% when losses offset — that is arithmetic, not a
  bug, and it is itself a warning sign.
- **Halves**: Welch t-test on first-half vs second-half mean.

**Verdicts:** `ROBUST`, `PERIOD_DEPENDENT`, `UNKNOWN`.

### `ml_scalper_v3` — PERIOD_DEPENDENT, decisively

```
n=54  total=$24,483  mean=$453.39  full-sample t=2.58
WORST leave-one-out: drop 2026-04 -> t=0.15, mean=$32.21
  period    n   drop_mean  drop_t   rest_mean  rest_t
  2026-03  10     -69.56   -0.17      572.24    2.99
  2026-04  29     816.48    3.22       32.21    0.15
  2026-05  15     100.06    0.42      589.29    2.64
top_5 = 62.1% of total P/L, top_10 = 96.4%
```

One month is the entire strategy. Outside April 2026 it earns $32/trade at t=0.15.

### `weekend_hold_v1` — the important one

This is the only leg in the ledger that survives the audited 16:45 ET mandatory
flatten, and it is carried in memory as "t=+4.47, DSR 0.997, 6/7 years positive".
Stressed (MNQ, 1 micro, net of costs, `results/edge_screens/weekend_hold_stress.json`):

```
n=202  total=$26,675  mean=$132.05  full-sample t=4.50
WORST leave-one-out (by year): drop 2026 -> t=3.38
  2020  n=30  mean=224.14  t= 3.39
  2021  n=31  mean= 26.99  t= 0.46
  2022  n=30  mean=-53.16  t=-0.69
  2023  n=30  mean= 94.38  t= 2.67
  2024  n=32  mean= 30.34  t= 0.54
  2025  n=32  mean=244.88  t= 2.65
  2026  n=17  mean=533.52  t= 3.94   (partial year)
top_10 = 43.5% of P/L, top_20 = 70.0%, gini 0.45
Halves: $57.54 -> $206.56  (Welch t=2.57, p=0.0108)
VERDICT: ROBUST (single-year LOO), but see below
```

**Read this carefully — the automated verdict is the least informative line.**
Single-year LOO says ROBUST only because dropping 1 of 7 years still leaves
~170 observations. Cutting the sample by hand tells a different story:

| sample | n | mean/wk | t |
|---|---:|---:|---:|
| all (2020–2026) | 202 | $132.05 | 4.50 |
| excl. 2026 (partial) | 185 | $95.16 | 3.38 |
| **excl. 2020 (COVID) and 2026** | 155 | $70.20 | **2.28** |
| **2021–2024 only (61% of sample)** | 123 | $24.75 | **0.84** |
| 2020–2022 | 91 | $65.56 | 1.62 |
| 2023–2026 | 111 | $186.56 | 4.51 |

"6/7 years positive" counts years at t=0.46 and t=0.54 as wins. The edge is a
flat middle (2021–2024) bracketed by two strong regimes. 2026 alone is 8.4% of
the weeks and 34% of the P/L.

This does not falsify the weekend premium — the mechanism (compensation for
carrying weekend gap risk) is sound and the placebos in the ledger are clean.
It does mean **the sizing decision cannot be made off t=4.50.**

### Vol-conditioning: tested and FALSIFIED

Hypothesis (from the "get paid for bearing risk" framing): if the premium
compensates gap risk, it should scale with prior-week realized vol. Tested with
expanding-window terciles (no lookahead), n=172:

```
HIGH vol (top tercile)    n= 34  mean=$142.08  t=1.22
MID  vol                  n= 57  mean=$137.03  t=2.51
LOW  vol (bottom tercile) n= 81  mean=$ 90.23  t=2.96
HIGH vol, 2021-2025 only  n= 28  mean=-$15.54  t=-0.14
```

High-vol weekends are **not** better — they are noisier and, excluding 2020 and
2026, slightly negative. The apparent high-vol strength is six weekends in 2026.
Conditioning on volatility does not improve this strategy. Do not re-test it.

### A caveat on the naive stop proxy

Flooring realized losses at −$300 raises the mean from $132 to $165 (+25%) and
caps the worst at −$300. **This is an upper bound, not a result.** It assumes
exit at exactly the stop and forgoes nothing on recovery. The punitive fill model
in `rule_based_v1/validation/research_disaster_stop.py` (stop fills at
`min(stop, bar_open)`, so a gap-through fills at the gap) finds the stop *costs*
premium: $135.51 → $103.57/week at 1 micro. Trust that one.
