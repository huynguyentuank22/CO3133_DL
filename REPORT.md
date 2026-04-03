| model_name | accuracy | macro_f1 | weighted_f1 | precision | recall | model_size_mb | inference_time_per_sample_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| **bert_llrd_weighted_ce** | **0.6825** | **0.5646** | **0.6891** | **0.5636** | **0.5690** | **417.66** | **6.7244** |    
| bert_full_undersample_ce | 0.6563 | 0.5490 | 0.6671 | 0.5296 | 0.5790 | 417.66 | 6.7301 |
| bert_full_weighted_ce | 0.6772 | 0.5471 | 0.6851 | 0.5632 | 0.5449 | 417.66 | 6.7322 |
| distilbert_full_weighted_ce | 0.6707 | 0.5421 | 0.6772 | 0.5352 | 0.5507 | 253.17 | 3.3773 |
| distilbert_llrd_weighted_ce | 0.6630 | 0.5409 | 0.6733 | 0.5453 | 0.5470 | 253.17 | 3.3800 |
| bert_llrd_undersample_ce | 0.6580 | 0.5247 | 0.6615 | 0.5190 | 0.5424 | 417.66 | 6.7320 |
| distilbert_llrd_undersample_ce | 0.6345 | 0.5205 | 0.6472 | 0.4996 | 0.5565 | 253.17 | 3.3803 |
| distilbert_full_undersample_ce | 0.6306 | 0.5181 | 0.6432 | 0.4971 | 0.5537 | 253.17 | 3.3802 |
| bilstm_attention_weighted_ce | 0.6356 | 0.4921 | 0.6421 | 0.5085 | 0.4860 | 15.25 | 0.5810 |
| bilstm_weighted_ce | 0.6524 | 0.4854 | 0.6511 | 0.4948 | 0.4783 | 15.25 | 0.5732 |
| bilstm_attention_undersample_ce | 0.5470 | 0.4354 | 0.5692 | 0.4351 | 0.4515 | 12.08 | 0.5802 |
| bilstm_undersample_ce | 0.5432 | 0.3825 | 0.5544 | 0.3779 | 0.4253 | 12.08 | 0.5741 |
| distilbert_freeze_weighted_ce | 0.5490 | 0.2826 | 0.5302 | 0.2901 | 0.2961 | 253.17 | 3.3637 |
| bert_freeze_weighted_ce | 0.4901 | 0.2525 | 0.4682 | 0.2601 | 0.2607 | 417.66 | 6.7240 |
| bert_freeze_undersample_ce | 0.3711 | 0.2267 | 0.4032 | 0.2363 | 0.2363 | 417.66 | 6.7224 |
| distilbert_freeze_undersample_ce | 0.4309 | 0.1944 | 0.4033 | 0.2634 | 0.2353 | 253.17 | 3.3674 |

### Best Model by Group (Macro-F1)

- **BiLSTM**: `bilstm_weighted_ce` (macro_f1 = 0.4854)
- **BiLSTM+Attention**: `bilstm_attention_weighted_ce` (macro_f1 = 0.4921)
- **DistilBERT**: `distilbert_full_weighted_ce` (macro_f1 = 0.5421)
- **BERT-base**: `bert_llrd_weighted_ce` (macro_f1 = 0.5646)

Overall best model: **`bert_llrd_weighted_ce`**.