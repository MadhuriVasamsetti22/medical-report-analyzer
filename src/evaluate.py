"""
Evaluation script — generates classification report, confusion matrix,
and summarization ROUGE scores.

Run after training:
    python src/evaluate.py
"""

import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    PROCESSED_CSV, CLF_SAVED_DIR, OUTPUTS_DIR,
    RANDOM_STATE, TEST_SIZE, VAL_SIZE, CLF_MAX_LEN,
)
from src.preprocess import build_datasets, basic_clean


def evaluate_classifier():
    print("Loading test set ...")
    _, _, test_ds, le = build_datasets(PROCESSED_CSV)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(CLF_SAVED_DIR))
    model     = AutoModelForSequenceClassification.from_pretrained(str(CLF_SAVED_DIR))
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    all_preds, all_labels = [], []

    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=16)

    print("Running inference on test set ...")
    for batch in loader:
        labels = batch.pop("labels").to(device)
        inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    print("\n── Classification Report ──────────────────────────────")
    print(classification_report(
        all_labels, all_preds, target_names=le.classes_, zero_division=0
    ))

    # Confusion matrix
    OUTPUTS_DIR.mkdir(exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm, xticklabels=le.classes_, yticklabels=le.classes_,
        annot=False, cmap="Blues", ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("BioClinicalBERT — Specialty Classification Confusion Matrix")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    out = OUTPUTS_DIR / "confusion_matrix_bert.png"
    plt.savefig(out, dpi=150)
    print(f"Confusion matrix saved → {out}")


if __name__ == "__main__":
    evaluate_classifier()