"""
Central config — all paths, hyperparameters, and model names live here.
Change anything once here and every module picks it up automatically.
"""
from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent
DATA_DIR      = ROOT_DIR / "data"
RAW_DATA_DIR  = DATA_DIR / "raw"
PROC_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR    = ROOT_DIR / "models"
OUTPUTS_DIR   = ROOT_DIR / "outputs"

# ── Dataset files ─────────────────────────────────────────────────────────────
RAW_CSV       = RAW_DATA_DIR  / "mtsamples.csv"
PROCESSED_CSV = PROC_DATA_DIR / "mtsamples_clean.csv"

# ── Column names (mtsamples dataset) ─────────────────────────────────────────
TEXT_COL      = "transcription"
LABEL_COL     = "medical_specialty"
DESC_COL      = "description"

# ── Preprocessing ─────────────────────────────────────────────────────────────
MIN_CLASS_SAMPLES = 30        # drop specialties with fewer than this many samples
RANDOM_STATE      = 42
TEST_SIZE         = 0.2
VAL_SIZE          = 0.1       # of training set

# ── DL Model — Classification ─────────────────────────────────────────────────
CLF_BASE_MODEL    = "emilyalsentzer/Bio_ClinicalBERT"   # pre-trained on clinical notes
CLF_MAX_LEN       = 512       # max tokens for BERT input
CLF_EPOCHS        = 4
CLF_BATCH_SIZE    = 8         # reduce to 4 if you hit OOM on GPU
CLF_LR            = 2e-5      # standard fine-tuning LR for BERT
CLF_WARMUP_RATIO  = 0.1
CLF_WEIGHT_DECAY  = 0.01
CLF_SAVED_DIR     = MODELS_DIR / "clinicalbert_specialty"

# ── DL Model — Summarization ──────────────────────────────────────────────────
SUM_BASE_MODEL    = "facebook/bart-large-cnn"
SUM_MAX_INPUT     = 1024
SUM_MAX_OUTPUT    = 180
SUM_MIN_OUTPUT    = 60

# ── NER — scispaCy ────────────────────────────────────────────────────────────
NER_MODEL         = "en_ner_bc5cdr_md"   # detects: DISEASE, CHEMICAL
SCI_MODEL         = "en_core_sci_md"     # general medical tokenizer

# ── Saved artefacts ───────────────────────────────────────────────────────────
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"