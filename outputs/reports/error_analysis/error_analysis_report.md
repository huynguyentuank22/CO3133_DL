# Error Analysis Report

## Top Confusion Pairs

| True Rating | Predicted Rating | Count |
|-------------|------------------|-------|
| 5 | 4 | 407 |
| 4 | 5 | 241 |
| 4 | 3 | 128 |
| 2 | 3 | 121 |
| 3 | 4 | 83 |

## Error Categories

| Category | Count |
|----------|-------|
| mixed_sentiment | 26 |
| subtle_rating_difference | 16 |
| ambiguous_review | 8 |

## Error Comparison: Weighted CE vs Undersample CE

| rating | total_samples | errors_weighted_ce | error_rate_weighted_ce | errors_undersample_ce | error_rate_undersample_ce |
|---|---|---|---|---|---|
| 1.0 | 123.0 | 56.0 | 0.4553 | 65.0 | 0.5285 |
| 2.0 | 232.0 | 206.0 | 0.8879 | 168.0 | 0.7241 |
| 3.0 | 424.0 | 174.0 | 0.4104 | 310.0 | 0.7311 |
| 4.0 | 736.0 | 382.0 | 0.519 | 553.0 | 0.7514 |
| 5.0 | 1880.0 | 455.0 | 0.242 | 344.0 | 0.183 |