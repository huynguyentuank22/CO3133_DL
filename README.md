# NLP Rating Classification — Women's E-Commerce Clothing Reviews

5-class rating classification (1–5) from review text, comparing **RNN** (BiLSTM, BiLSTM+Attention) vs **Transformer** (DistilBERT, BERT-base) models.

## Features

- **4 Models**: BiLSTM, BiLSTM+Attention, DistilBERT, BERT-base
- **Imbalance Strategies**: Weighted CE vs Undersampled CE
- **Fine-tuning Strategies**: Freeze, Full, Layer-wise LR Decay (LLRD)
- **XAI**: Attention visualization, Captum Integrated Gradients, LIME
- **Error Analysis**: Confusion pairs, sample categorization, cross-strategy comparison
- **Augmentation**: Synonym replacement + random deletion
- **Robustness**: Noisy test set evaluation (typos, case, punctuation)
- **Efficiency**: Parameters, model size, inference time comparison
- **Ensemble**: Weighted probability averaging (RNN + Transformer)
- **Demo**: Streamlit web app

![alt text](image.png)

## Project Structure

```
├── data/raw/               # Raw CSV dataset
├── data/splits/            # Train/Val/Test CSVs
├── src/                    # Core modules (17 files)
├── scripts/                # CLI scripts (14 files)
├── app/                    # Streamlit demo
├── outputs/                # Checkpoints, figures, reports, tables
├── notebooks/              # Jupyter notebooks for analysis
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place dataset in data/raw/
# File: "Womens Clothing E-Commerce Reviews.csv"

# 3. Prepare data (clean, split, save)
python scripts/prepare_data.py
```

## Training Models

```bash
# BiLSTM
python scripts/train_bilstm.py --imbalance weighted_ce --epochs 15

# BiLSTM + Attention
python scripts/train_bilstm_attention.py --imbalance weighted_ce --epochs 15

# DistilBERT (with fine-tune strategy)
python scripts/train_distilbert.py --finetune full --imbalance weighted_ce --epochs 4

# BERT-base
python scripts/train_transformer_large.py --finetune full --imbalance weighted_ce --epochs 4
```

### Fine-tuning Strategies (Transformers)

```bash
python scripts/train_distilbert.py --finetune freeze   # Head only
python scripts/train_distilbert.py --finetune full      # Full fine-tune
python scripts/train_distilbert.py --finetune llrd      # Layer-wise LR decay
```

## Evaluation

```bash
# Evaluate all trained models on test set
python scripts/evaluate_all.py
# → outputs/tables/model_comparison.csv
```

## Latest Model Comparison Results

Source: `outputs/tables/model_comparison.csv`

| model_name | accuracy | macro_f1 | weighted_f1 | precision | recall | model_size_mb | inference_time_per_sample_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bilstm_weighted_ce | 0.6074 | 0.4758 | 0.6183 | 0.4623 | 0.4948 | 15.25 | 1.1469 |
| bilstm_undersample_ce | 0.5432 | 0.3825 | 0.5544 | 0.3779 | 0.4253 | 12.08 | 1.0841 |
| bilstm_attention_weighted_ce | 0.6477 | 0.5022 | 0.6525 | 0.4948 | 0.5134 | 15.25 | 1.1822 |
| bilstm_attention_undersample_ce | 0.5758 | 0.3882 | 0.5653 | 0.3857 | 0.4164 | 12.08 | 1.1243 |
| distilbert_freeze_weighted_ce | 0.5490 | 0.2826 | 0.5302 | 0.2901 | 0.2961 | 253.17 | 18.4477 |
| distilbert_freeze_undersample_ce | 0.4309 | 0.1944 | 0.4033 | 0.2634 | 0.2353 | 253.17 | 22.0174 |
| distilbert_full_weighted_ce | 0.6592 | 0.5392 | 0.6703 | 0.5509 | 0.5433 | 253.17 | 22.1121 |
| distilbert_full_undersample_ce | 0.6306 | 0.5181 | 0.6432 | 0.4971 | 0.5537 | 253.17 | 24.6578 |
| distilbert_llrd_weighted_ce | 0.6604 | 0.5428 | 0.6714 | 0.5413 | 0.5528 | 253.17 | 25.9265 |
| distilbert_llrd_undersample_ce | 0.6295 | 0.5114 | 0.6410 | 0.4903 | 0.5495 | 253.17 | 30.8181 |
| bert_freeze_weighted_ce | 0.4901 | 0.2525 | 0.4682 | 0.2601 | 0.2607 | 417.66 | 58.1530 |
| bert_freeze_undersample_ce | 0.3711 | 0.2267 | 0.4032 | 0.2363 | 0.2363 | 417.66 | 59.8117 |
| bert_full_weighted_ce | 0.6630 | 0.5474 | 0.6765 | 0.5579 | 0.5525 | 417.66 | 65.6044 |
| bert_full_undersample_ce | 0.6574 | 0.5391 | 0.6646 | 0.5207 | 0.5740 | 417.66 | 63.3480 |
| bert_llrd_weighted_ce | 0.6825 | 0.5646 | 0.6891 | 0.5636 | 0.5690 | 417.66 | 58.7579 |
| bert_llrd_undersample_ce | 0.6580 | 0.5247 | 0.6615 | 0.5190 | 0.5424 | 417.66 | 60.6179 |

## Imbalance Strategy Comparison

```bash
# Compare weighted_ce vs undersample_ce
python scripts/run_imbalance_comparison.py
# → outputs/tables/imbalance_comparison.csv
# → outputs/figures/class_distribution_comparison.png
```

## XAI / Interpretability

```bash
python scripts/run_xai.py
# → outputs/reports/xai_results/*.html
```

## Error Analysis

```bash
python scripts/run_error_analysis.py
# → outputs/reports/error_analysis/
```

## Augmentation & Robustness

```bash
python scripts/run_augmentation.py     # With/without augmentation
python scripts/run_robustness.py       # Clean vs noisy test set
```

## Efficiency Comparison

```bash
python scripts/run_efficiency.py
# → outputs/tables/efficiency_comparison.csv
```

## Ensemble

```bash
python scripts/run_ensemble.py
# → outputs/tables/ensemble_comparison.csv
```

## Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

## Hardware Notes

- **GPU recommended** for Transformer training (CUDA)
- **CPU** sufficient for RNN training and inference
- Estimated VRAM: ~4GB for DistilBERT, ~6GB for BERT-base (batch_size=16)
- Training times (approx): RNN ~10min, DistilBERT ~30min, BERT-base ~1h (on GPU)

## Dataset

- **Source**: Women's E-Commerce Clothing Reviews
- **Samples**: ~23,000 after cleaning
- **Input**: Title + " [SEP] " + Review Text
- **Target**: Rating (1–5) → mapped to labels (0–4)
- **Split**: 70% train / 15% val / 15% test (stratified)
