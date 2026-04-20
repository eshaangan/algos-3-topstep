# Deferred integrations (plan scope)

The following GitHub directions are **not** integrated into this repo; revisit only if data or goals change.

## Order-book / HFT ([ML-HFT](https://github.com/bradleyboyuyang/ML-HFT))

Requires Level-2 (depth) or tick data and a different feature pipeline than the current **5m RTH CUSUM** MES stack. No code paths were added.

## HMM + LSTM academic stack ([Stock-Market-Trend-Analysis-Using-HMM-LSTM](https://github.com/JINGEWU/Stock-Market-Trend-Analysis-Using-HMM-LSTM))

Daily China A-share workflow with hybrid sequence models. Porting would need a new labeling horizon, leakage-safe CV, and execution alignment with triple-barrier events. Use the lightweight **hmmlearn** feature experiment ([README_HMM_EXPERIMENT.md](README_HMM_EXPERIMENT.md)) first.

## MlFinLab ([mlfinlab](https://github.com/hudson-and-thames/mlfinlab))

Public repository is issue-tracking; the library is commercial. Patterns from *Advances in Financial Machine Learning* are already approximated here (e.g. sample decay, triple-barrier labels, meta filtering).

## quant-traderr-lab ([quant-traderr-lab](https://github.com/quant-traderr/quant-traderr-lab))

Notebook-style references only; no dependency or vendored code.
