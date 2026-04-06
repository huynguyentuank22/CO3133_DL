# NLP Rating Classification - Womens Clothing E-Commerce Reviews

End-to-end 5-class rating classification (1 to 5) from customer review text.
This repository includes:
- data preprocessing and stratified train/val/test splitting,
- RNN and Transformer training pipelines,
- evaluation and comparison across checkpoints,
- error analysis, XAI, robustness checks, and ensembling,
- interactive demos with Streamlit and Gradio (HF Space folder).

## Current Scope

- Model families:
  - BiLSTM
  - BiLSTM + Attention
  - DistilBERT
  - BERT-base
- Imbalance strategies:
  - weighted_ce
  - undersample_ce
- Transformer fine-tuning strategies:
  - freeze
  - full
  - llrd
- Analysis pipelines:
  - evaluate all checkpoints
  - error analysis (best models)
  - XAI (IG/LIME)
  - robustness (clean vs noisy)
  - weighted-probability ensemble

## Latest Snapshot (from outputs/tables/model_comparison.csv)

- Best overall model: bert_llrd_weighted_ce
  - Accuracy: 0.6825
  - Macro-F1: 0.5646
  - Weighted-F1: 0.6891
  - MAE: 0.3505
- Best-by-family (Macro-F1):
  - BiLSTM: bilstm_weighted_ce (0.4854)
  - BiLSTM+Attention: bilstm_attention_weighted_ce (0.4921)
  - DistilBERT: distilbert_full_weighted_ce (0.5421)
  - BERT-base: bert_llrd_weighted_ce (0.5646)

## Repository Layout

```text
.
|- app/
|  |- streamlit_app.py
|- data/
|  |- raw/
|  |  `- Womens Clothing E-Commerce Reviews.csv
|  |- splits/
|  |  |- train.csv
|  |  |- val.csv
|  |  `- test.csv
|  |- processed/
|  |  |- vocab.json
|  |  `- vocabs/
|  `- wandb-scerets/
|- hf_space/
|  |- app.py
|  |- README.md
|  |- requirements.txt
|  `- data/processed/vocabs/
|- notebooks/
|  `- eda.ipynb
|- outputs/
|  |- checkpoints/
|  |- figures/
|  |- logs/
|  |- reports/
|  `- tables/
|- scripts/
|  |- prepare_data.py
|  |- train_bilstm.py
|  |- train_bilstm_attention.py
|  |- train_distilbert.py
|  |- train_transformer_large.py
|  |- evaluate_all.py
|  |- run_error_analysis.py
|  |- run_xai.py
|  |- run_robustness.py
|  `- run_ensemble.py
|- src/
|  |- preprocessing.py
|  |- datasets.py
|  |- trainer.py
|  |- metrics.py
|  |- xai_utils.py
|  `- ...
|- web_report/
|  |- index.html
|  |- styles.css
|  `- script.js
|- kaggle_runner.py
|- REPORT.md
|- requirements.txt
`- README.md
```

## Environment Setup

Recommended Python version: 3.10+

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data Preparation

The preprocessing pipeline expects the raw CSV at:
- data/raw/Womens Clothing E-Commerce Reviews.csv

Run:

```bash
python scripts/prepare_data.py
```

Outputs:
- data/splits/train.csv
- data/splits/val.csv
- data/splits/test.csv

Note:
- If split files already exist and you want to keep them, skip prepare_data.py.

## Training

### RNN

```bash
# BiLSTM
python scripts/train_bilstm.py --imbalance weighted_ce --epochs 15

# BiLSTM + Attention
python scripts/train_bilstm_attention.py --imbalance weighted_ce --epochs 15
```

### Transformer

```bash
# DistilBERT
python scripts/train_distilbert.py --finetune full --imbalance weighted_ce --epochs 4

# BERT-base
python scripts/train_transformer_large.py --finetune llrd --imbalance weighted_ce --epochs 4
```

### Common CLI Arguments

- --imbalance: weighted_ce | undersample_ce
- --epochs: number of training epochs
- --batch_size: batch size
- --lr: learning rate
- --wandb: force enable W&B logging
- --wandb_project, --wandb_entity, --wandb_run_name: W&B metadata
- --finetune (transformers only): freeze | full | llrd

Checkpoints are saved to:
- outputs/checkpoints/*_best.pt

## Evaluation and Analysis

### 1) Evaluate all checkpoints

```bash
python scripts/evaluate_all.py
```

Main outputs:
- outputs/tables/model_comparison.csv
- outputs/reports/*_metrics.json
- outputs/figures/*_confusion_matrix.png

### 2) Error analysis (best models)

```bash
python scripts/run_error_analysis.py
```

Outputs:
- outputs/reports/error_analysis/error_analysis_report.md
- outputs/reports/error_analysis/error_summary_best_models.csv
- outputs/reports/error_analysis/*_misclassified.csv

### 3) XAI (best models)

```bash
python scripts/run_xai.py
```

Outputs:
- outputs/reports/xai_results/*.html

### 4) Robustness (clean vs noisy)

```bash
python scripts/run_robustness.py
```

Outputs:
- outputs/tables/robustness_results.csv

### 5) Ensemble

```bash
python scripts/run_ensemble.py
```

Outputs:
- outputs/tables/ensemble_comparison.csv
- outputs/tables/ensemble_alpha_search.csv

Important:
- The model pair for ensembling is currently hard-coded in scripts/run_ensemble.py (MODEL_A and MODEL_B).

## Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

The app auto-detects available checkpoints and loads only models that have both:
- checkpoint file in outputs/checkpoints/
- matching vocab file for RNN models in data/processed/vocabs/

## Hugging Face Space (Gradio) Folder

Local run (inside this repository):

```bash
pip install -r hf_space/requirements.txt
python hf_space/app.py
```

Notes:
- hf_space/app.py downloads model files from the hard-coded Hugging Face model repo:
  - huynguyentuan/DL-assignment-1

## Weights and Biases (Optional)

You can enable W&B by either:
- passing --wandb to train scripts, or
- using env-based settings consumed by src/utils.py.

A sample secret/env location exists at:
- data/wandb-scerets/wandb.env

## Kaggle Runner (Optional)

For Kaggle workflow automation, use:
- kaggle_runner.py

Typical command:

```bash
kaggle kernels push -p .
```

## Related Reports

- Full experiment report: REPORT.md
- Web report assets: web_report/

## Practical Notes

- RNN checkpoints require matching vocab files in data/processed/vocabs/.
- outputs/ already contains generated artifacts and pretrained checkpoints in this repository state.
- If you retrain models, regenerate evaluation and analysis outputs to keep tables/reports consistent.
