# Backtest reports

Walk-forward (expanding-window) backtest. Lower is better for both metrics.

## Overall (pooled across folds)

| Model | n | Log loss | Brier |
|---|---|---|---|
| baseline_elo | 760 | 1.0159 | 0.6089 |
| rf | 760 | 1.0324 | 0.6189 |
| log_reg | 760 | 1.0807 | 0.6396 |

## By test season

| Model | Season | n | Log loss | Brier |
|---|---|---|---|---|
| baseline_elo | 2024 | 380 | 1.0037 | 0.6003 |
| baseline_elo | 2025 | 380 | 1.0282 | 0.6175 |
| log_reg | 2024 | 380 | 1.0858 | 0.6391 |
| log_reg | 2025 | 380 | 1.0756 | 0.6400 |
| rf | 2024 | 380 | 1.0210 | 0.6094 |
| rf | 2025 | 380 | 1.0438 | 0.6284 |
