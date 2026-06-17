# Model Design Rationale

Why the match predictor is built the way it is, and the evidence behind each decision. The
product outputs probabilities (→ implied odds), so every model is judged on **log loss** and
**Brier score**, not raw accuracy — accuracy throws away the probability quality that matters.

## Baselines (the bars to beat)

- **Elo** (`baseline_elo`): a sequential rating that updates after every match. It is a strong,
  simple, well-calibrated prior and the benchmark the ML models must beat.
- **Multinomial Logistic Regression** (`log_reg`): a fast, interpretable linear ML baseline,
  scaled in a pipeline. It establishes whether a linear model can extract signal at all.

Both exist to contextualise the main model and to guard against shipping something that a
trivial approach already beats.

## Main model: Random Forest

Random Forest is the project's chosen main model: it captures non-linear interactions between
form, Elo, and league position, needs no feature scaling, and exposes feature importances for
explainability (feeding later SHAP work).

**Walk-forward backtest (760 matches, lower is better):**

| Model | Log loss | Brier |
|---|---|---|
| baseline_elo | **1.0159** | **0.6089** |
| rf_tuned | 1.0238 | 0.6137 |
| rf (default) | 1.0324 | 0.6189 |
| log_reg | 1.0807 | 0.6396 |

Reference points: a constant base-rate predictor scores **1.0798** and a uniform guess **1.0986**,
so the entire skill envelope is only ~0.064 — football outcomes are highly random.

## Diagnostics (notebook `03_rf_diagnostics.ipynb`)

- The RF largely **rediscovers Elo** (Elo is the strongest feature block) and initially trailed it.
- **Draws are structurally unlearnable** from the current features (recall ≈ 0.04; no feature
  signals a draw) — a feature limitation, not a tuning one.
- The original "overconfidence" hypothesis was **rejected**: the RF is no more extreme than Elo;
  its weakness is under-concentrating probability on the true class.
- A `max_depth` sweep showed fully-grown trees (`None`) are the wrong default; **depth ≈ 4** is optimal.

## Tuning

`RandomizedSearchCV` + `TimeSeriesSplit(5)` on `neg_log_loss`, with an **expanded** grid
(shallow depths added on the sweep evidence). Best params: `max_depth=4`,
`min_samples_leaf=10`, `n_estimators=300`, `max_features=log2`. This improved the RF from
1.0324 → **1.0238** and regularised it (no more overfitting). Honest caveat: the gain is **not
statistically significant** (Wilcoxon p≈0.49) and still trails Elo — tuning closes most of the
gap but does not overtake the baseline.

## Recency weighting — tried and rejected

We tested exponential `sample_weight` decay to counter recent-season weakness. A half-life
sweep (120–1095 days) made log loss **worse** at every meaningful decay; only "effectively no
decay" matched the tuned model. Cause: ~3 seasons is too little data to down-weight, and Elo
already tracks evolving strength. The served model stays **unweighted**; the code is kept as
documented evidence.

## Conclusion

The tuned Random Forest is the main model per the project design. It is competitive with Elo
and near the practical ceiling for pre-match features. The highest-impact remaining lever is
**draw-signal feature engineering**, not further model tuning.
