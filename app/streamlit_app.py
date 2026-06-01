"""
Streamlit demo — full DL pipeline.
Run: streamlit run app/streamlit_app.py
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Medical Report Analyzer",
    page_icon="🏥",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏥 Medical Report Analyzer")
st.caption("AI-powered · Non-diagnostic · Educational use only")
st.warning(
    "**⚠️ Disclaimer:** This tool does NOT provide medical diagnoses or advice. "
    "All outputs are AI-generated for informational and educational purposes only. "
    "Always consult a licensed healthcare professional.",
    icon="⚠️",
)
st.divider()


# ── Model loading (cached so it only loads once) ──────────────────────────────
@st.cache_resource(show_spinner="Loading AI models — this takes ~30s the first time ...")
def load_analyzer():
    from src.predict import ReportAnalyzer
    return ReportAnalyzer(load_summarizer=True, load_ner=True)


# ── Input ─────────────────────────────────────────────────────────────────────
col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.subheader("Input")
    report = st.text_area(
        "Paste medical transcription",
        value="",
        height=380,
        placeholder="Paste any medical transcription here and click Analyze Report...",
        label_visibility="collapsed",
    )
    run = st.button("🔍 Analyze Report", type="primary", use_container_width=True)

# ── Output ────────────────────────────────────────────────────────────────────
with col_out:
    st.subheader("Analysis")

    if run:
        if not report.strip():
            st.warning("Please paste a transcription first.")
        else:
            try:
                analyzer = load_analyzer()

                # ── 1. Classification ─────────────────────────────────────
                with st.spinner("Classifying specialty ..."):
                    clf_result = analyzer.classify(report)

                st.markdown("**📋 Specialty Classification**")
                m1, m2 = st.columns(2)
                m1.metric("Predicted Specialty", clf_result["top_specialty"])
                m2.metric("Confidence",          f"{clf_result['confidence']}%")

                for item in clf_result["top_3"]:
                    st.progress(
                        item["confidence"] / 100,
                        text=f"{item['specialty']} — {item['confidence']}%"
                    )

                st.divider()

                # ── 2. NER ────────────────────────────────────────────────
                with st.spinner("Extracting medical entities ..."):
                    entities = analyzer.extract(report)

                st.markdown("**🔬 Extracted Entities**")
                ec1, ec2 = st.columns(2)

                diseases  = entities.get("DISEASE",  [])
                chemicals = entities.get("CHEMICAL", [])

                with ec1:
                    st.markdown("**Conditions / Diseases**")
                    if diseases:
                        for e in diseases[:10]:
                            st.markdown(f"- {e['text']}")
                    else:
                        st.caption("None detected")

                with ec2:
                    st.markdown("**Medications / Chemicals**")
                    if chemicals:
                        for e in chemicals[:10]:
                            st.markdown(f"- {e['text']}")
                    else:
                        st.caption("None detected")

                st.divider()

                # ── 3. Patient Summary ────────────────────────────────────
                with st.spinner("Generating patient-friendly summary ..."):
                    sum_result = analyzer.summarize_report(report)

                st.markdown("**💬 Patient-Friendly Summary**")
                st.info(sum_result["summary"])
                st.caption(sum_result["disclaimer"])

            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.caption("Results will appear here after clicking Analyze.")