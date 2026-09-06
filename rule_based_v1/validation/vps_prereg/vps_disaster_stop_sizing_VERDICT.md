# vps_disaster_stop_sizing — the stop WORKS, and it still does not reach the bar

Scripts `research_disaster_stop.py`, `research_required_edge.py`.
Runs `runs/disaster_stop{,_104,_phidias}.json`, `runs/required_edge.json`.

## The hypothesis was right: the premium is not in the deep excursions

A hard stop caps the tail while leaving the mean almost intact. Per micro, weekend leg,
n=202, with gap-through fills (stop fills at `min(stop, bar open)`):

| stop/micro | mean $ | t | worst $ | stop rate | mean kept | tail cut |
|---|---|---|---|---|---|---|
| none | 135.51 | 4.58 | −1,117 | 0% | 100% | 0% |
| $300 | 103.57 | 3.83 | −311 | 32.2% | 76% | 72% |
| $500 | 114.34 | 4.05 | −503 | 14.4% | 84% | 55% |
| **$600** | **126.60** | **4.36** | **−603** | **8.9%** | **93%** | **46%** |
| $1,000 | 124.53 | 4.13 | −1,003 | 2.5% | 92% | 10% |

**A $600/micro stop keeps 93% of the premium and cuts the worst trade by 46%**, with
t rising to 4.36 and only 8.9% of weekends stopped. The FOMC leg behaves the same way
($500 stop keeps 101% of the mean). This is a real, free risk improvement.

## A bug this run exposed

The first pass carried the **unstopped** path minimum into the MC's intra-trade bust
check — charging the account for excursions a stopped position was never in, which made
every stopped configuration look untradeable. `run_leg()` now truncates the path at the
stop bar. Anyone reusing this must keep that: a stop changes the MAE, not just the exit.

## Free win at the long horizon

Best stopped config beats the unstopped book on every axis simultaneously:

| config | P(pass) | median wks | decay ×0.50 | worst weekend |
|---|---|---|---|---|
| unstopped, wk×1 + fomc×2 | 92.0% | 33 | 70.9% | −$1,117 (37% of MLL) |
| **$600 stop, wk×1 + fomc×2** | **94.4%** | **35** | **73.2%** | **−$603 (20% of MLL)** |

Higher P(pass), more decay-robust, and a worst trade cut from 37% to 20% of the MLL, for
two extra weeks. **Adopt the $600 stop regardless of what else is decided.**

## But it does not unlock the deadline

The stop lets larger sizes clear the 60%-of-MLL tail budget — $300 stop at 5 micros earns
**$585/week**, comfortably above the $375–500 the goal's arithmetic asked for, with worst
weekend −$1,554 (52% of MLL). P(pass) at 16 weeks is still only **36%**.

P(pass) as a function of size is single-peaked and never approaches the bar:

| weekend × | $/wk | worst weekend | P(pass) @16wk | tail legal |
|---|---|---|---|---|
| 1 | 162 | −$603 | 5.7% | yes |
| 2 | 324 | −$1,205 | 41.3% | no |
| **3** | **486** | −$1,808 | **48.6% (peak)** | no |
| 5 | 740 | −$3,014 | 35.7% | no |
| 8 | 1,119 | −$4,822 | 22.5% | no |

## The quantified NO-GO

Holding dispersion and the tail fixed and scaling only the mean — i.e. asking for a
*better edge*, not a bigger bet — at the largest tail-legal size:

| edge quality | $/wk | P(pass) @16wk |
|---|---|---|
| ×1.0 (actual) | 162 | 5.7% |
| ×2.0 | 324 | 34.4% |
| ×2.5 | 405 | 59.1% |
| ×3.0 | 486 | 80.5% |
| **×3.2 (interpolated)** | **~520** | **~85%** |
| ×4.0 | 649 | 98.4% |

**85% in 16 weeks requires roughly a 3.2× better edge at the same risk — about
$520/week versus the $162/week the book earns at the only tail-legal size.**

## This corrects the goal's own arithmetic

The goal states the target "needs roughly $375–500 per week." That is a *mean* condition
and it understates the requirement, because it ignores path risk. At $405/week the MC
gives **59%**, not 85%. The honest figure is **~$520/week at the ×1 tail**, or more if
the extra return is bought with size rather than with edge quality — because size scales
the tail too, and the tail is what busts the account.

**Verdict: the stop is adopted as a risk improvement. The 16-week bar is not reachable
with this book, and the gap is a factor of ~3.2 in edge quality, not a tuning problem.**
