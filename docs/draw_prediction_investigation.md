# Draw Prediction Investigation & Model Acceptance

**Date:** 2026-06-18 · **Notebooks:** `05_data_and_model_explore.ipynb`, `06_draw_diagnostic.ipynb`

## Motivation

The tuned Random Forest (`rf_tuned`) predicted **0 of 197 draws** in the walk-forward
backtest. This session investigated whether that was a data fault, a model fault, or a
genuine limit — and decided whether to accept the current model.

## What we did

1. **Data readiness EDA** on the exact matrix the model consumes (`build_features`). Data is
   clean: 1140 rows / 3 seasons, 0 missing values, schema matches the locked 34 features.
   Distributions, Elo trajectories, and class balance all sensible.
2. **Draw mechanism diagnostic.** Measured `p_draw` and the draw-class one-vs-rest **AUC**
   across every model.

## Key findings

- **It is a feature ceiling, not a data or model fault.** Draw-class AUC ≈ **0.50** for *every*
  model (Elo, logistic regression, RF, tuned RF, calibrated RF) — coin-flip. `p_draw` sits at
  roughly the base rate (~0.24) but is identical on draws vs non-draws (0.237 vs 0.234): the
  model spreads draw probability sensibly but cannot identify *which* matches draw.
- **"0 draws" is an argmax artifact.** `p_draw` is never suppressed to zero; it is simply never
  the highest class. Models forced to pick more draws (`log_reg`: 59) had **worse** log loss —
  their draw picks are noise.
- This confirms the Sprint 2 conclusion (recency weighting, draw-signal features, and
  calibration were all tried and reverted as negative).

## Options considered & rejected

| Option | Verdict |
|---|---|
| Re-tune / class weights / thresholds | Rejected — buys fake draw recall at the cost of log loss. |
| **xG features** | Rejected — Football-Data.org has no xG at any tier; a live source would be a second weekly dependency, and xG is post-match (leakage/availability issues). |
| **More seasons** | Rejected — free API tier returns `403` for 2022; older data is paywalled. |
| **Odds as a feature** | Rejected — odds are paywalled on our API and unavailable for upcoming fixtures. Training on historical-only odds causes **train/serve skew** (the model relies on a feature it can't see at inference), likely degrading 2026 performance. |

## Decision

**Accept the current `rf_tuned` model.** Draws are near-unpredictable from pre-match tabular
features — a structural property of football, not a defect. The model beats naive baselines,
is well-calibrated, and stays clean and single-source (one API, no live external dependency).

## Future work (out of scope)

- **Benchmark vs market odds** (free historical odds, analysis-only) to quantify how close the
  model is to the predictive ceiling and whether even the market can rank draws.
- Revisit xG/odds *as features* only if a reliable live data source is adopted.
