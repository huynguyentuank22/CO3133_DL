# NLP Rating Classification - Women's E-Commerce Clothing Reviews

End-to-end 5-class rating classification (1-5) from review text.
The project compares RNN-based and Transformer-based models, then adds error analysis,
XAI, robustness testing, ensemble evaluation, and a Streamlit demo.

## Current Scope

- 4 model families:
	- BiLSTM
	- BiLSTM + Attention
	- DistilBERT
	- BERT-base
- Imbalance strategies: `weighted_ce`, `undersample_ce`
- Transformer fine-tuning strategies: `freeze`, `full`, `llrd`
- Analysis pipelines:
	- Full evaluation across all checkpoints
	- Error analysis (best models)
	- XAI (IG/LIME)
	- Robustness (clean vs noisy)
	- Two-model weighted ensemble
- Streamlit demo for interactive inference

## Repository Layout

```text
.
|- app/
|  |- streamlit_app.py
|- data/
|  |- raw/
|  |- splits/
|  |- processed/
|- scripts/
|  |- prepare_data.py
|  |- train_bilstm.py
|  |- train_bilstm_attention.py
|  |- train_distilbert.py
|  |- train_transformer_large.py
|  |- evaluate_all.py
|  |- run_xai.py
|  |- run_error_analysis.py
|  |- run_robustness.py
|  |- run_ensemble.py
|- src/
|- outputs/
|  |- checkpoints/
|  |- figures/
|  |- reports/
|  |- tables/
|  |- logs/
|- kaggle_runner.py
|- REPORT.md
|- requirements.txt
`- README.md
```

## Installation

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Data Preparation

1. Put dataset CSV at:
	 - `data/raw/Womens Clothing E-Commerce Reviews.csv`
2. Run preprocessing:

```bash
python scripts/prepare_data.py
```

Generated splits:
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

## Training

### RNN Models

```bash
# BiLSTM
python scripts/train_bilstm.py --imbalance weighted_ce --epochs 15

# BiLSTM + Attention
python scripts/train_bilstm_attention.py --imbalance weighted_ce --epochs 15
```

### Transformer Models

```bash
# DistilBERT
python scripts/train_distilbert.py --finetune full --imbalance weighted_ce --epochs 4

# BERT-base
python scripts/train_transformer_large.py --finetune llrd --imbalance weighted_ce --epochs 4
```

### Common Training Arguments

- `--imbalance`: `weighted_ce` or `undersample_ce`
- `--epochs`: number of epochs
- `--batch_size`: batch size
- `--lr`: learning rate
- `--wandb`: enable Weights and Biases logging
- `--wandb_project`, `--wandb_entity`, `--wandb_run_name`: W&B options

Transformer-only argument:
- `--finetune`: `freeze`, `full`, `llrd`

## Evaluation and Analysis

### 1) Evaluate all checkpoints

```bash
python scripts/evaluate_all.py
```

Primary outputs:
- `outputs/tables/model_comparison.csv`
- confusion matrices (`*_confusion_matrix.png`) under `outputs/figures/`
- per-model metrics JSON under `outputs/reports/`

### 2) Error analysis (best models only)

```bash
python scripts/run_error_analysis.py
```

Outputs:
- `outputs/reports/error_analysis/error_analysis_report.md`
- `outputs/reports/error_analysis/error_summary_best_models.csv`
- `outputs/reports/error_analysis/*_misclassified.csv`

### 3) XAI (best models only)

```bash
python scripts/run_xai.py
```

Outputs:
- `outputs/reports/xai_results/*.html`

### 4) Robustness (clean vs noisy)

```bash
python scripts/run_robustness.py
```

Outputs:
- `outputs/tables/robustness_results.csv`

### 5) Ensemble

```bash
python scripts/run_ensemble.py
```

Outputs:
- `outputs/tables/ensemble_comparison.csv`
- `outputs/tables/ensemble_alpha_search.csv`

Note:
- `run_ensemble.py` currently ensembles two hard-coded checkpoints (edit `MODEL_A` / `MODEL_B` inside the script to change pair).

## Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

The app auto-detects available checkpoints and supports explanation methods per model.

## Current Best Snapshot

From latest evaluation artifacts in `outputs/tables/model_comparison.csv`:

- Best overall: `bert_llrd_weighted_ce`
	- Accuracy: `0.6825`
	- Macro-F1: `0.5646`
	- Weighted-F1: `0.6891`

Best-by-family macro-F1:
- BiLSTM: `bilstm_weighted_ce` (`0.4854`)
- BiLSTM+Attention: `bilstm_attention_weighted_ce` (`0.4921`)
- DistilBERT: `distilbert_full_weighted_ce` (`0.5421`)
- BERT-base: `bert_llrd_weighted_ce` (`0.5646`)

See full report in `REPORT.md`.

## Kaggle Runner (Optional)

If you use Kaggle kernels, this project includes `kaggle_runner.py` to automate
training/evaluation steps with Kaggle datasets and secrets.

Typical push command:

```bash
kaggle kernels push -p .
```

## Notes

- There is currently no standalone `scripts/run_augmentation.py` in this repository.
- Augmentation helpers exist in `src/augmentation.py` but are not exposed as a separate CLI script.
