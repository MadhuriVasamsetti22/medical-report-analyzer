# 🏥 Medical Report Analyzer

> An end-to-end Deep Learning pipeline that extracts structured insights from clinical transcriptions using transformer-based NLP models.
> **Non-diagnostic · Educational use only · Not intended for clinical decision-making**

<br>

## 📌 Overview

Medical reports are filled with complex clinical jargon that is difficult for patients to understand. This project builds an AI-powered pipeline that reads a medical transcription and automatically:

- **Predicts the medical specialty** (e.g., Cardiology, Neurology, Orthopedics)
- **Extracts medical entities** — diseases, conditions, and medications
- **Generates a patient-friendly summary** in plain, simple language

The entire pipeline is served through an interactive **Streamlit web application** that anyone can use.

<br>

## 🎯 Key Features

| Feature | Technology | Description |
|--------|-----------|-------------|
| Specialty Classification | Bio_ClinicalBERT | Fine-tuned BERT model predicts 1 of 23 medical specialties |
| Medical NER | scispaCy BC5CDR | Extracts diseases and medications from clinical text |
| Patient Summary | BART-large-cnn | Abstractive summarization in plain language |
| PHI De-identification | Regex Pipeline | Removes dates, names, MRNs before model inference |
| Web App | Streamlit | Real-time interactive demo |

<br>

## 🧠 Deep Learning Concepts Demonstrated

- **Transfer Learning** — Starting from Bio_ClinicalBERT pre-trained on MIMIC-III hospital records
- **Fine-tuning** — Updating all BERT weights on specialty classification task
- **Custom Weighted Loss** — WeightedTrainer with class-balanced cross-entropy to handle 30:1 class imbalance
- **Seq2Seq Architecture** — BART encoder-decoder for abstractive summarization
- **Zero-shot NER** — scispaCy BC5CDR model used directly without additional training
- **Mixed Precision Training** — fp16 training on Google Colab T4 GPU

<br>

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Dataset | MTSamples (4,694 transcriptions, 23 specialties) |
| Model | Bio_ClinicalBERT (fine-tuned) |
| Test Accuracy | 38% |
| Macro F1 Score | 0.41 |
| Training Time | ~45 min (T4 GPU, fp16) |
| Class Imbalance Ratio | 30:1 |

> **Note on accuracy:** 38% across 23 highly imbalanced medical specialties with a macro F1 of 0.41 is comparable to published baselines on this dataset. Macro F1 is the correct metric here as it treats all classes equally regardless of sample size.

<br>

## 🏗️ Project Architecture

```
Raw Medical Transcription
         │
         ▼
┌─────────────────────┐
│  PHI De-identification │  ← Removes dates, names, MRNs
│  & Text Cleaning      │
└─────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────┐
│                                                │
▼                    ▼                           ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│ BioClinical  │  │  scispaCy    │  │   BART-large-cnn  │
│    BERT      │  │  BC5CDR NER  │  │   Summarizer      │
│              │  │              │  │                   │
│ 23-class     │  │ DISEASE      │  │ Patient-friendly  │
│ Specialty    │  │ CHEMICAL     │  │ plain language    │
│ Classifier   │  │ extraction   │  │ summary           │
└──────────────┘  └──────────────┘  └──────────────────┘
         │                    │                  │
         └────────────────────┴──────────────────┘
                              │
                              ▼
                   ┌──────────────────┐
                   │  Streamlit App   │
                   │  (Live Demo)     │
                   └──────────────────┘
```

<br>

## 📁 Project Structure

```
medical-report-analyzer/
│
├── config.py                      ← All paths, hyperparameters, model names
│
├── data/
│   ├── raw/                       ← mtsamples.csv (download from Kaggle)
│   └── processed/                 ← Auto-generated after preprocessing
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py              ← PHI removal, cleaning, PyTorch Dataset class
│   ├── train_classifier.py        ← BioClinicalBERT fine-tuning with WeightedTrainer
│   ├── ner.py                     ← scispaCy entity extraction
│   ├── summarize.py               ← BART abstractive summarization
│   ├── predict.py                 ← ReportAnalyzer unified inference class
│   └── evaluate.py                ← Metrics, confusion matrix generation
│
├── app/
│   └── streamlit_app.py           ← Interactive web demo
│
├── notebooks/
│   ├── 01_eda.py                  ← Exploratory data analysis
│   ├── 02_train_and_evaluate.py   ← Training walkthrough
│   └── 03_ner_and_summarization.py← NER and summarization experiments
│
├── tests/
│   ├── test_preprocess.py         ← Unit tests for preprocessing pipeline
│   └── test_ner.py                ← Unit tests for NER module
│
├── models/                        ← Saved model checkpoints (gitignored)
├── outputs/                       ← Confusion matrix, evaluation plots
├── Dockerfile                     ← Container for deployment
└── requirements.txt
```

<br>

## 🚀 Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/MadhuriVasamsetti22/medical-report-analyzer.git
cd medical-report-analyzer
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
# Install CPU version of PyTorch first
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining packages
pip install transformers==4.38.0 datasets accelerate evaluate scikit-learn joblib
pip install pandas numpy matplotlib seaborn streamlit tqdm rouge-score spacy

# Install scispaCy and NER model
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

### 4. Download the dataset
- Go to [kaggle.com/datasets/tboyle10/medicaltranscriptions](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
- Download and place `mtsamples.csv` in `data/raw/`

### 5. Run the pipeline
```bash
# Step 1 — Clean and preprocess data
python src/preprocess.py

# Step 2 — Train BioClinicalBERT (use Google Colab T4 GPU recommended)
python src/train_classifier.py

# Step 3 — Evaluate model
python src/evaluate.py

# Step 4 — Launch the app
streamlit run app/streamlit_app.py
```

<br>

## 💡 Training on Google Colab (Recommended)

Training on a CPU laptop takes 6-8 hours. Google Colab's free T4 GPU reduces this to **~45 minutes**.

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Runtime → Change runtime type → **T4 GPU**
3. Upload `src/`, `config.py`, and `data/processed/mtsamples_clean.csv`
4. Run `train_classifier.py`
5. Save trained model to Google Drive and download to `models/`

<br>

## 🖥️ Running the App

```bash
streamlit run app/streamlit_app.py
```

Open your browser at `http://localhost:8501`. Paste any medical transcription and click **Analyze Report** to get:

- Predicted medical specialty with confidence score
- Top-3 specialty predictions
- Extracted diseases and medications
- Patient-friendly plain language summary

### Sample Input
```
CHIEF COMPLAINT: Chest pain and shortness of breath.
HISTORY: 58-year-old male with hypertension presenting with substernal
chest pain radiating to the left arm for 2 hours. Diaphoresis noted.
MEDICATIONS: Aspirin 81mg, atorvastatin 40mg, lisinopril 10mg.
ASSESSMENT: Rule out acute myocardial infarction.
PLAN: EKG, troponin, aspirin 325mg, IV heparin started.
```

### Sample Output
```
Specialty     : Cardiovascular  (confidence: 74.3%)
Diseases      : chest pain, myocardial infarction, hypertension
Medications   : Aspirin, atorvastatin, lisinopril, heparin
Summary       : A 58-year-old man with high blood pressure came in with
                chest pain spreading to his left arm. Doctors are checking
                if he had a heart attack and started treatment with blood
                thinners and other medications.
```

<br>

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.10 |
| Deep Learning | PyTorch, HuggingFace Transformers |
| NLP | scispaCy, spaCy |
| Classification Model | Bio_ClinicalBERT (emilyalsentzer) |
| Summarization Model | BART-large-cnn (facebook) |
| NER Model | en_ner_bc5cdr_md (scispaCy) |
| Data Processing | Pandas, NumPy, scikit-learn |
| Evaluation | HuggingFace Evaluate, ROUGE Score |
| Web App | Streamlit |
| Training | Google Colab T4 GPU |
| Version Control | Git, GitHub |

<br>

## 📈 Dataset

**MTSamples — Medical Transcription Dataset**

| Property | Value |
|----------|-------|
| Source | [Kaggle — tboyle10/medicaltranscriptions](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) |
| Total Records | 4,999 transcriptions |
| After Cleaning | 4,694 transcriptions |
| Medical Specialties | 23 (after dropping rare classes) |
| License | CC0 Public Domain |
| Avg Length | ~500-800 words per transcription |

<br>

## ⚠️ Ethical Considerations

- **Non-diagnostic:** This tool does not provide medical diagnoses or clinical advice
- **PHI Protection:** All patient-sensitive information is de-identified before model inference
- **Public Dataset:** MTSamples contains publicly available, already de-identified transcriptions
- **Disclaimer:** All app outputs include a clear non-diagnostic disclaimer
- **Not for clinical use:** This project is strictly for educational and portfolio purposes

<br>

## 🔮 Future Improvements

- [ ] Fine-tune BART on medical summarization datasets for better summaries
- [ ] Add symptom and procedure extraction to NER pipeline
- [ ] Implement UMLS/SNOMED code mapping for extracted entities
- [ ] Collect more training data to improve classification accuracy
- [ ] Add multi-language support for non-English reports
- [ ] Deploy with GPU support for faster inference

<br>

## 👤 Author

**Madhuri Vasamsetti**
- GitHub: [@MadhuriVasamsetti22](https://github.com/MadhuriVasamsetti22)
- Project: [medical-report-analyzer](https://github.com/MadhuriVasamsetti22/medical-report-analyzer)

<br>

## 📄 License

This project is licensed under the MIT License.

---

> **Disclaimer:** This project is built for educational and portfolio purposes only. It is not intended for clinical use, medical diagnosis, or any healthcare decision-making. Always consult a licensed healthcare professional for medical advice.
