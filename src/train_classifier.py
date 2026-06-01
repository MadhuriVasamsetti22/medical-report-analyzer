"""
Fine-tune Bio_ClinicalBERT for medical specialty classification.

DL concepts demonstrated here:
  - Transfer learning: start from a pre-trained BERT, fine-tune on our task
  - AutoModelForSequenceClassification: adds a linear head on top of BERT
  - HuggingFace Trainer: handles training loop, gradient accumulation, checkpointing
  - class_weights: combat the severe class imbalance in mtsamples
  - compute_metrics: evaluate macro-F1 after each epoch

Run:
    python src/train_classifier.py
"""

import sys, json
import numpy as np
import torch
from pathlib import Path
from collections import Counter

import evaluate
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    PROCESSED_CSV, CLF_BASE_MODEL, CLF_EPOCHS, CLF_BATCH_SIZE,
    CLF_LR, CLF_WARMUP_RATIO, CLF_WEIGHT_DECAY,
    CLF_SAVED_DIR, MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE,
)
from src.preprocess import build_datasets


# ── Weighted loss to handle class imbalance ───────────────────────────────────
class WeightedTrainer(Trainer):
    """
    Custom Trainer that applies class weights to the cross-entropy loss.

    WHY: mtsamples has ~1000 surgery samples but only ~30 for rare specialties.
    Without weighting, the model just predicts 'surgery' for everything.
    Class weights penalise mistakes on rare classes more heavily.
    """

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights   # Tensor of shape (num_classes,)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        # Weighted cross-entropy
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Metrics ───────────────────────────────────────────────────────────────────
accuracy_metric = evaluate.load("accuracy")
f1_metric       = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc   = accuracy_metric.compute(predictions=preds, references=labels)
    f1    = f1_metric.compute(predictions=preds, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "macro_f1": f1["f1"]}


# ── Main training function ────────────────────────────────────────────────────
def train():
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    import joblib
    from config import LABEL_ENCODER_PATH

    print("=" * 60)
    print("  Bio_ClinicalBERT — Medical Specialty Classifier")
    print("=" * 60)

    # 1. Build datasets
    train_ds, val_ds, test_ds, le = build_datasets(PROCESSED_CSV)
    num_labels = len(le.classes_)

    # 2. Compute class weights from training labels
    train_labels = [sample["labels"].item() for sample in train_ds]
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_labels),
        y=train_labels,
    )
    class_weights = torch.tensor(cw, dtype=torch.float32)
    print(f"\nNum classes    : {num_labels}")
    print(f"Class weight min/max: {cw.min():.2f} / {cw.max():.2f}")

    # 3. Load pre-trained model + add classification head
    print(f"\nLoading base model: {CLF_BASE_MODEL}")
    model = AutoModelForSequenceClassification.from_pretrained(
        CLF_BASE_MODEL,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,   # the classification head is new
    )

    # 4. Training arguments
    #    Every argument is explained so you know what you're setting
    args = TrainingArguments(
        output_dir=str(CLF_SAVED_DIR),

        # ── Epochs & batch ─────────────────────────────────────────────────
        num_train_epochs=CLF_EPOCHS,
        per_device_train_batch_size=CLF_BATCH_SIZE,
        per_device_eval_batch_size=CLF_BATCH_SIZE * 2,

        # ── Optimiser ──────────────────────────────────────────────────────
        learning_rate=CLF_LR,
        weight_decay=CLF_WEIGHT_DECAY,       # L2 regularisation
        warmup_ratio=CLF_WARMUP_RATIO,       # ramp up LR for first 10% of steps
        lr_scheduler_type="linear",

        # ── Evaluation & saving ────────────────────────────────────────────
        eval_strategy="epoch",               # evaluate after every epoch
        save_strategy="epoch",
        load_best_model_at_end=True,         # restore best checkpoint at end
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        # ── Logging ────────────────────────────────────────────────────────
        logging_dir=str(OUTPUTS_DIR / "logs"),
        logging_steps=50,
        report_to="none",                    # set to "wandb" if you want tracking

        # ── Reproducibility ────────────────────────────────────────────────
        seed=RANDOM_STATE,

        # ── GPU memory ─────────────────────────────────────────────────────
        fp16=torch.cuda.is_available(),      # mixed precision if GPU available
    )

    # 5. Instantiate weighted trainer
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 6. Train
    print("\nStarting training ...")
    trainer.train()

    # 7. Evaluate on held-out test set
    print("\n── Test Set Evaluation ──────────────────────────────────")
    test_results = trainer.evaluate(test_ds)
    print(test_results)

    # 8. Save final model + tokenizer
    CLF_SAVED_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(CLF_SAVED_DIR))
    AutoTokenizer.from_pretrained(CLF_BASE_MODEL).save_pretrained(str(CLF_SAVED_DIR))

    # Save num_labels so predict.py can load the model correctly
    with open(CLF_SAVED_DIR / "label_config.json", "w") as f:
        json.dump({"num_labels": num_labels, "classes": le.classes_.tolist()}, f)

    print(f"\nModel saved → {CLF_SAVED_DIR}")
    return test_results


if __name__ == "__main__":
    train()