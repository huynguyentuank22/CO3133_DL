# Error Analysis Report (Best Models)


## bilstm (bilstm_weighted_ce)

### Top Confusion Pairs

| True Rating | Predicted Rating | Count |
|-------------|------------------|-------|
| 4 | 5 | 283 |
| 5 | 4 | 279 |
| 3 | 4 | 109 |
| 2 | 3 | 95 |
| 4 | 3 | 90 |
| 3 | 2 | 72 |
| 1 | 2 | 47 |
| 5 | 3 | 33 |
| 2 | 1 | 29 |
| 2 | 4 | 29 |

### Error Categories

| Category | Count |
|----------|-------|
| subtle_rating_difference | 47 |
| mixed_sentiment | 2 |
| ambiguous_review | 1 |

## bilstm_attn (bilstm_attention_weighted_ce)

### Top Confusion Pairs

| True Rating | Predicted Rating | Count |
|-------------|------------------|-------|
| 5 | 4 | 395 |
| 4 | 5 | 244 |
| 3 | 4 | 109 |
| 2 | 3 | 95 |
| 3 | 2 | 88 |
| 4 | 3 | 77 |
| 1 | 2 | 55 |
| 3 | 5 | 31 |
| 5 | 3 | 27 |
| 2 | 1 | 25 |

### Error Categories

| Category | Count |
|----------|-------|
| subtle_rating_difference | 50 |

## distilbert (distilbert_full_weighted_ce)

### Top Confusion Pairs

| True Rating | Predicted Rating | Count |
|-------------|------------------|-------|
| 5 | 4 | 317 |
| 4 | 5 | 220 |
| 3 | 2 | 117 |
| 4 | 3 | 116 |
| 2 | 3 | 90 |
| 3 | 4 | 63 |
| 1 | 2 | 41 |
| 2 | 1 | 38 |
| 5 | 3 | 35 |
| 3 | 1 | 20 |

### Error Categories

| Category | Count |
|----------|-------|
| subtle_rating_difference | 41 |
| mixed_sentiment | 9 |

## bert (bert_llrd_weighted_ce)

### Top Confusion Pairs

| True Rating | Predicted Rating | Count |
|-------------|------------------|-------|
| 5 | 4 | 327 |
| 4 | 5 | 211 |
| 3 | 2 | 107 |
| 4 | 3 | 107 |
| 2 | 3 | 80 |
| 3 | 4 | 70 |
| 1 | 2 | 45 |
| 2 | 1 | 30 |
| 5 | 3 | 25 |
| 3 | 1 | 16 |

### Error Categories

| Category | Count |
|----------|-------|
| subtle_rating_difference | 45 |
| mixed_sentiment | 4 |
| ambiguous_review | 1 |

## Summary (Best Models)

| Model Family | Checkpoint | Accuracy | Error Rate |
|--------------|------------|----------|------------|
| bilstm | bilstm_weighted_ce | 0.6524 | 0.3476 |
| bilstm_attn | bilstm_attention_weighted_ce | 0.6356 | 0.3644 |
| distilbert | distilbert_full_weighted_ce | 0.6707 | 0.3293 |
| bert | bert_llrd_weighted_ce | 0.6825 | 0.3175 |