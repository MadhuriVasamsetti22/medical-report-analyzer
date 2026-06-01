"""
Preprocessing pipeline + PyTorch Dataset class for BERT fine-tuning.

Two jobs:
  1. clean_and_save()  — run once to produce data/processed/mtsamples_clean.csv
  2. MedicalDataset    — PyTorch Dataset used by the Trainer in train_classifier.py
"""

import re
import sys
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import joblib

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    RAW_CSV, PROCESSED_CSV, TEXT_COL, LABEL_COL,
    MIN_CLASS_SAMPLES, CLF_BASE_MODEL, CLF_MAX_LEN,
    LABEL_ENCODER_PATH, MODELS_DIR,
)

# ── PHI de-identification ─────────────────────────────────────────────────────
_PHI = [
    (r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',                     '[DATE]'),
    (r'\b(?:Dr|MD|DO)\.?\s+[A-Z][a-z]+ ?[A-Z]?[a-z]*',  '[PHYSICIAN]'),
    (r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b',                 '[PHONE]'),
    (r'\bMR#?\s*\d+\b',                                   '[MRN]'),
    (r'\b\d{5}(?:-\d{4})?\b',                             '[ZIP]'),
]


def deidentify(text: str) -> str:
    """Remove common PHI patterns before any model sees the text."""
    for pattern, tag in _PHI:
        text = re.sub(pattern, tag, text)
    return text


def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = deidentify(text)
    text = re.sub(r'\s+', ' ', text)   # collapse whitespace
    return text.strip()


def clean_label(label: str) -> str:
    if not isinstance(label, str):
        return "unknown"
    return label.strip().lower().replace('/', '_').replace(' ', '_')


# ── Step 1: run once ──────────────────────────────────────────────────────────
def clean_and_save(csv_path=RAW_CSV, out_path=PROCESSED_CSV):
    """
    Load raw mtsamples.csv → clean → drop rare specialties → save.
    Run: python src/preprocess.py
    """
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Raw shape: {df.shape}")

    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df = df[df[TEXT_COL].str.strip() != ""]

    df["clean_text"]  = df[TEXT_COL].apply(basic_clean)
    df["clean_label"] = df[LABEL_COL].apply(clean_label)

    # Drop specialties with too few samples to learn from
    counts = df["clean_label"].value_counts()
    valid  = counts[counts >= MIN_CLASS_SAMPLES].index
    df     = df[df["clean_label"].isin(valid)].reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"  Clean shape : {df.shape}")
    print(f"  Specialties : {df['clean_label'].nunique()}")
    print(f"  Saved → {out_path}")
    return df


# ── Step 2: PyTorch Dataset for BERT ─────────────────────────────────────────
class MedicalDataset(Dataset):
    """
    Wraps tokenized text + integer labels for use with HuggingFace Trainer.

    What's happening here (DL concept):
      - BERT needs text converted to token IDs + attention masks
      - AutoTokenizer handles this — it's the BERT vocabulary lookup
      - We return a dict with input_ids, attention_mask, labels
        which is exactly what BERT's forward() expects
    """

    def __init__(self, texts: list, labels: list, tokenizer, max_len: int = CLF_MAX_LEN):
        self.tokenizer = tokenizer
        self.texts     = texts
        self.labels    = labels
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,        # long reports get cut at 512 tokens
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),       # shape: (512,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # shape: (512,)
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def build_datasets(csv_path=PROCESSED_CSV):
    """
    Load processed CSV → encode labels → tokenize → return train/val/test datasets.
    Called by train_classifier.py.
    """
    from sklearn.model_selection import train_test_split
    from config import RANDOM_STATE, TEST_SIZE, VAL_SIZE

    df = pd.read_csv(csv_path)
    texts  = df["clean_text"].tolist()
    labels = df["clean_label"].tolist()

    # Integer-encode the specialty strings
    le = LabelEncoder()
    y  = le.fit_transform(labels)

    # Save label encoder so predict.py can decode integers back to names
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"Label encoder saved → {LABEL_ENCODER_PATH}")
    print(f"Classes: {len(le.classes_)}")

    # Stratified splits — keeps class distribution balanced across splits
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE,
        random_state=RANDOM_STATE, stratify=y_train
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    tokenizer = AutoTokenizer.from_pretrained(CLF_BASE_MODEL)

    train_ds = MedicalDataset(X_train, y_train.tolist(), tokenizer)
    val_ds   = MedicalDataset(X_val,   y_val.tolist(),   tokenizer)
    test_ds  = MedicalDataset(X_test,  y_test.tolist(),  tokenizer)

    return train_ds, val_ds, test_ds, le


if __name__ == "__main__":
    clean_and_save()