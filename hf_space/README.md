---
title: CO3133 Rating Predictor (DistilBERT)
emoji: "⭐"
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 4.44.1
python_version: 3.10.13
app_file: app.py
pinned: false
preload_from_hub:
  - huynguyentuan/DL-assignment-1 distilbert_full_weighted_ce_best.pt
---

# Gradio Space: Multi-Model Rating Predictor

This Space supports these model options (same as Streamlit demo):
- `BERT-base` (`bert_llrd_weighted_ce_best.pt`) + IG
- `DistilBERT` (`distilbert_full_weighted_ce_best.pt`) + IG
- `BiLSTM + Attention` (`bilstm_attention_weighted_ce_best.pt`) + IG
- `BiLSTM` (`bilstm_weighted_ce_best.pt`) + LIME

## Model Source (Hard-Coded)

The app currently hard-codes this value in `app.py`:

- `MODEL_REPO_ID = "huynguyentuan/DL-assignment-1"`

## Notes

- The app reuses existing project modules from `src/` (inference + IG/LIME explanation).
- If you create a separate Space repository, copy `src/` (and any local package files it depends on) into that Space repo.
