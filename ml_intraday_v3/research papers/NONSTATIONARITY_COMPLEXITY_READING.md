# Nonstationarity vs. Model Complexity — reading notes and one missing PDF

Added 2026-09-08 while mining the Quant Guild channel/library for anything
applicable to this repo. Both entries below address the same failure mode our
own ledger keeps reproducing: complex models fit to an intraday tape whose
data-generating process moves between train and test.

## 1. MISSING — needs a manual download (SSRN is gated)

> Capponi, A., Huang, C., Sidaoui, J. A., Wang, K., and Zou, J. (2025).
> *The Nonstationarity-Complexity Tradeoff in Return Prediction*.
> SSRN working paper 5980654, December 28, 2025.
> https://ssrn.com/abstract=5980654 — DOI 10.2139/ssrn.5980654

Surfaced via Quant Guild lecture 77 ("Profitable vs Tradable"), where Paolucci
cites it as his former Columbia stochastics/time-series group's treatment of
exactly the problem that kills retail-scale strategies.

**Not in this folder.** SSRN returns HTTP 403 to scripted fetches (Cloudflare)
and the paper is not on arXiv or OpenAlex — it is SSRN-only. It needs a browser
download. Please drop the PDF in this folder when convenient.

Why it is worth having: every model in `ml_intraday_v3/` is a complexity choice
made without an explicit stationarity budget. This paper is the formal statement
of the tradeoff we have been paying for implicitly.

## 2. PRESENT — freely available and on the same mechanism

> Coqueret, G. and Laguerre, M. (2026). *Overparametrized models with
> posterior drift*. arXiv:2506.23619 (submitted Jun 2025, revised May 2026).
> File: `Overparametrized models with posterior drift - 2506.23619.pdf`

Abstract (verbatim):

> This paper investigates the impact of posterior drift on out-of-sample
> forecasting accuracy in overparametrized machine learning models. We document
> the loss in performance when the loadings of the data generating process
> change between the training and testing samples. This matters crucially in
> settings in which regime changes are likely to occur, for instance, in
> financial markets. Applied to equity premium forecasting, our results
> underline the sensitivity of a market timing strategy to sub-periods and to
> the bandwidth parameters that control the complexity of the model. For the
> average investor, we find that focusing on holding periods of 15 years can
> generate very heterogeneous returns, especially for small bandwidths. Large
> bandwidths yield much more consistent outcomes, but are far less appealing
> from a risk-adjusted return standpoint. All in all, our findings tend to
> recommend cautiousness when resorting to large linear models for stock market
> predictions.

### Relevance to this repo

The paper's core result is a *tradeoff*, not a fix: the complexity settings that
look best on risk-adjusted return are the same ones whose out-of-sample outcomes
are most heterogeneous across sub-periods. Restated for us — the configuration
that wins a sweep is selected partly for its sensitivity to the sub-period it
was swept on, which is a mechanism for the dev-to-holdout collapses recorded
throughout the ledger (e.g. `range_ignite_v1`, dev DSR 0.34 -> holdout -$3,003).

Caveat before anyone over-reads it: their setting is equity-premium forecasting
at multi-year holding periods with linear/kernel models, not 5-minute futures
barriers with gradient boosting. The *mechanism* (posterior drift between train
and test) transfers; the specific bandwidth findings do not.

### Related tooling in this repo

`ml_intraday_v3/analysis/stability.py` is the empirical counterpart to this
literature: it measures whether the realized P/L distribution actually drifted,
which is the observable consequence of the posterior drift these papers model.
See `EDGE_SCREENS_GUIDE.md`.
