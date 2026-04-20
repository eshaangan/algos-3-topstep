# Quant-Traderr Transfer Map

This repo review treats `quant-traderr-lab` as a **concept library**, not a production dependency.

## Keep

1. `Hidden Markov`
- Best fit: optional regime features for standalone ML.
- Local hook: `ml_intraday_v3/configs/features.yaml` already exposes `hmm_regime`.
- Why: potentially useful as a compact state feature, but only after the standalone pipeline proves directionality and OOS stability.

2. `Wavelet Transform`
- Best fit: optional multi-resolution features for standalone ML.
- Local hook: `ml_intraday_v3/configs/features.yaml` already exposes `multi_resolution`.
- Why: good as a compact volatility/scale experiment, not as a core fix.

3. `Shannon Entropy`
- Best fit: optional compression/uncertainty features for standalone ML.
- Local hook: add later through `ml_intraday_v3/features/build.py` once the current standalone candidate clears hard gates.
- Why: lightweight regime-state feature, but not important enough to come before execution alignment.

4. `Hawkes Process`
- Best fit: research-only order-flow clustering idea.
- Local hook: can be explored later if real order-flow inputs become available.
- Why: conceptually interesting, but current repo mostly has bar-based proxies.

## Defer

1. `RMT_Correlation_Filter`
- Weak fit for the current single-instrument intraday setup.
- Better suited to cross-asset or portfolio models than MES/MNQ bar prediction.

2. `Neural Network`
- The external repo's NN script is mostly a training visualization.
- It does not address the repo's actual standalone-ML problems: direction bias, regime instability, and Topstep-safe OOS performance.

## Discard As Non-Transferable

1. PNG / MP4 rendering logic
- No trading edge.
- High maintenance cost.

2. Standalone one-file "pipeline" structure
- This repo already has a stronger architecture for labels, backtesting, walk-forward, and risk.
- Porting that style would be a regression in reproducibility.

## Current Priority Order

1. Make standalone ML execution-aligned and Topstep-gated.
2. Prove balanced long/short behavior across multiple OOS windows.
3. Only then test optional imported concepts such as `hmm_regime` or `multi_resolution`.
