"""
Medical Named Entity Recognition using scispaCy.

Extracts:
  DISEASE  — conditions, diagnoses (e.g. "myocardial infarction", "diabetes")
  CHEMICAL — drugs, medications  (e.g. "aspirin", "metformin")

Model: en_ner_bc5cdr_md (trained on BioCreative V CDR corpus)

Install before using:
    pip install scispacy
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

Run standalone:
    python src/ner.py
"""

import sys
from pathlib import Path
from typing import List, Dict

sys.path.append(str(Path(__file__).parent.parent))
from config import NER_MODEL


# ── Lazy loader — model is large, only load when needed ──────────────────────
_nlp_ner = None


def _get_ner_model():
    global _nlp_ner
    if _nlp_ner is None:
        try:
            import spacy
            _nlp_ner = spacy.load(NER_MODEL)
            print(f"NER model loaded: {NER_MODEL}")
        except OSError:
            raise OSError(
                f"scispaCy model '{NER_MODEL}' not found.\n"
                "Install it with:\n"
                "  pip install scispacy\n"
                "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
                "releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
            )
    return _nlp_ner


# ── Core extraction function ──────────────────────────────────────────────────
def extract_entities(text: str) -> Dict[str, List[Dict]]:
    """
    Run NER on medical text and return structured entity dict.

    Returns:
        {
          "DISEASE":  [{"text": "chest pain", "start": 12, "end": 22}, ...],
          "CHEMICAL": [{"text": "aspirin",    "start": 55, "end": 62}, ...]
        }

    DL concept: scispaCy uses a CNN-based NER model under the hood.
    The model was trained on the BC5CDR corpus (chemical-disease relations).
    We're doing zero-shot inference — no training needed on our side.
    """
    nlp = _get_ner_model()
    doc = nlp(text[:100_000])   # cap at 100k chars for memory safety

    result: Dict[str, List] = {"DISEASE": [], "CHEMICAL": []}

    seen = set()   # deduplicate by (text, label)
    for ent in doc.ents:
        key = (ent.text.lower(), ent.label_)
        if key not in seen and ent.label_ in result:
            seen.add(key)
            result[ent.label_].append({
                "text":  ent.text,
                "start": ent.start_char,
                "end":   ent.end_char,
                "label": ent.label_,
            })

    return result


def entity_summary(entities: Dict[str, List[Dict]]) -> str:
    """
    Human-readable one-liner of what was extracted.
    Shown in the Streamlit app below the entity table.
    """
    diseases  = [e["text"] for e in entities.get("DISEASE",  [])]
    chemicals = [e["text"] for e in entities.get("CHEMICAL", [])]
    parts = []
    if diseases:
        parts.append(f"Conditions: {', '.join(diseases[:5])}")
    if chemicals:
        parts.append(f"Medications: {', '.join(chemicals[:5])}")
    return " | ".join(parts) if parts else "No medical entities detected."


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = (
        "Patient has a history of hypertension and type 2 diabetes mellitus. "
        "Currently on metformin 500mg twice daily and lisinopril 10mg. "
        "Presenting with chest pain and shortness of breath."
    )
    ents = extract_entities(sample)
    print("DISEASES  :", [e["text"] for e in ents["DISEASE"]])
    print("CHEMICALS :", [e["text"] for e in ents["CHEMICAL"]])
    print("\nSummary:", entity_summary(ents))