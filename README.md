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

<!-- ## Imbalance Strategy Comparison

```bash
# Compare weighted_ce vs undersample_ce
python scripts/run_imbalance_comparison.py
# → outputs/tables/imbalance_comparison.csv
# → outputs/figures/class_distribution_comparison.png
``` -->

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
