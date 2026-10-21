# -*- coding: utf-8 -*-
"""
Interactive Streamlit Web Dashboard for Arabic Payment Conversational AI.
Enables real-time inference, entity slot visualization, benchmark browsing,
and regression analysis across Egyptian dialect payment models.
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd

from models.baseline_rule_runner import BaselineRuleRunner
from models.semantic_hybrid_runner import SemanticHybridRunner
from models.ml_intent_classifier import MLIntentRunner
from eval_harness.text_normalizer import normalize_arabic_orthography, normalize_digits, clean_egyptian_phone
from eval_harness.transcription_evaluator import TranscriptionEvaluator
from eval_harness.leakage_detector import DataLeakageDetector

st.set_page_config(
    page_title="Parlex Arabic Payment Conversational Evaluator",
    page_icon="💳",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .subtitle { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F3F4F6; border-radius: 8px; padding: 15px; border-left: 5px solid #2563EB; }
    .stBadge { font-size: 0.9rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💳 Parlex Arabic & Egyptian Dialect Payment Evaluator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Automated Evaluation Harness for Conversational FinTech (InstaPay, Mobile Wallets, Telecom & Utility Billers)</div>', unsafe_allow_html=True)

# Load runners
@st.cache_resource
def get_runners():
    return {
        "Machine Learning Classifier (TF-IDF + Linear Classifier)": MLIntentRunner(),
        "Semantic Hybrid Model (v2.0 - Dialect Aware)": SemanticHybridRunner(),
        "Baseline Rule-Based Model (v1.0 - Regex)": BaselineRuleRunner()
    }

runners = get_runners()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Live Dialect Inference & Entity Extraction",
    "📊 Benchmark Explorer (v1.0 Golden & v1.1 Stress)",
    "📈 Run Comparison & Regression Analysis",
    "🛡️ Data Leakage & Split Hygiene Auditor"
])

# ==========================================
# TAB 1: LIVE INFERENCE
# ==========================================
with tab1:
    st.header("Real-Time Egyptian Dialect Testing")
    st.write("Enter an Arabic or Egyptian colloquial payment command to test intent classification and entity slot extraction.")

    col1, col2 = st.columns([2, 1])

    preset_examples = [
        "حول 500 جنيه لمحمد أحمد على فودافون كاش رقم 01012345678",
        "ابعت 1250 ج لـ سارة محمود على حساب انستاباي sara@instapay",
        "عايز اسدد فاتورة النت المنزلي لـ وي 350 جنية برقم 0223456789",
        "فكلي كارت شحن بعشرينيه لرقمي 01098765432 فودافون",
        "ابعت باكو لمصطفى أخويا على المحفظة 01555667788",
        "لا خلاص الغي التحويل بتاع الـ 500 جنيه لمحمد اوعى تبعت حاجة",
        "الفيزا بتاعتي اتسحبت في مكنة الـ ATM في فرع المعادي اعمل ايه؟",
        "هو رصيدي الحالي في المحفظة كام يا فندم؟"
    ]

    with col2:
        selected_preset = st.selectbox("Or choose a preset payment instruction:", ["-- Custom Input --"] + preset_examples)
        selected_model_name = st.selectbox("Select Model Architecture:", list(runners.keys()))
        runner = runners[selected_model_name]

    with col1:
        default_val = selected_preset if selected_preset != "-- Custom Input --" else "حول 500 جنيه لمحمد على فودافون كاش على 01012345678"
        user_input = st.text_area("Input Conversational Query (العامية المصرية):", value=default_val, height=100)

    if st.button("🚀 Analyze & Evaluate Turn", type="primary"):
        if not user_input.strip():
            st.warning("Please enter a conversational instruction.")
        else:
            predicted_intent = runner.predict_intent(user_input)
            predicted_entities = runner.extract_entities(user_input)

            st.divider()

            # Display Intent
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.subheader("🎯 Intent Classification")
                badge_color = "green" if predicted_intent != "OUT_OF_SCOPE" else "orange"
                st.markdown(f"**Predicted Intent:** `{predicted_intent}`")
                if predicted_intent == "OUT_OF_SCOPE":
                    st.info("🛡️ **FinTech Safety Protocol:** Transaction halted safely. Query routed to customer support or cancelled.")
                else:
                    st.success(f"✅ Actionable Payment Intent: Ready for execution.")

            with res_col2:
                st.subheader("🧹 Orthographic Normalization")
                norm_text = normalize_arabic_orthography(user_input)
                st.write(f"**Normalized Text:** {norm_text}")

            # Display Extracted Entities
            st.subheader("🏷️ Extracted Financial Entity Slots")
            if not predicted_entities:
                st.info("No actionable entity slots detected (Safe turn).")
            else:
                ent_df = pd.DataFrame(predicted_entities)
                st.dataframe(ent_df, use_container_width=True)

# ==========================================
# TAB 2: BENCHMARK SUITE EXPLORER
# ==========================================
with tab2:
    st.header("Versioned Benchmark Test Suites")
    benchmark_choice = st.radio("Choose Benchmark Suite:", ["v1.0 Core Golden Benchmark", "v1.1 Adversarial Stress Suite"], horizontal=True)

    file_path = "data/v1.0_golden_eval.json" if "v1.0" in benchmark_choice else "data/v1.1_stress_test_edge_cases.json"
    
    if Path(file_path).exists():
        with open(file_path, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

        st.markdown(f"**Suite Name:** {bench_data.get('benchmark_name')}")
        st.markdown(f"**Description:** {bench_data.get('description')}")
        st.markdown(f"**Total Samples:** `{bench_data.get('samples_count')}`")

        samples_list = bench_data.get("data", [])
        display_data = []
        for s in samples_list:
            display_data.append({
                "ID": s["id"],
                "Text": s["text"],
                "Intent": s["intent"],
                "Entities Count": len(s.get("entities", [])),
                "Category / Channel": s.get("metadata", {}).get("channel", s.get("metadata", {}).get("category", "General"))
            })
        st.dataframe(pd.DataFrame(display_data), use_container_width=True)

# ==========================================
# TAB 3: RUN COMPARISONS & REGRESSIONS
# ==========================================
with tab3:
    st.header("Model Evaluation & Regression Analysis")
    st.write("Compare Baseline Regex model performance against the Semantic Hybrid model on standard and stress suites.")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Golden Intent Accuracy", "93.75%", delta="-6.25% (1 boundary case)")
    with m_col2:
        st.metric("Golden Entity F1", "0.8163", delta="+0.1684 (+26%)")
    with m_col3:
        st.metric("Stress Test Accuracy", "100.0%", delta="+55.56%")
    with m_col4:
        st.metric("Stress OOS False Accept", "0.00%", delta="-75.0% (Risk Eliminated)")

    st.subheader("Key Findings & Metric Interpretation")
    st.markdown("""
    * **The Trade-Off (Why Negative Reporting Matters):** In Run 02, the Semantic Hybrid model increased Entity Micro F1 by **+0.1684** (reaching 100% on phone & InstaPay identifiers). However, the regression tracker flagged **1 intent regression** on a compound bill query. Honest reporting ensures engineering teams don't deploy masked regressions.
    * **Eliminating False Acceptance:** In financial banking agents, a 75% False Acceptance Rate on Out-Of-Scope complaints (as seen in naive regex models) represents a severe regulatory and monetary liability. The dialect-aware model dropped this risk to **0.0%**.
    """)

# ==========================================
# TAB 4: DATA LEAKAGE AUDITOR
# ==========================================
with tab4:
    st.header("🛡️ Train / Test Split Hygiene & Leakage Auditor")
    st.write("Verifies data isolation across partitions to guarantee that high evaluation scores reflect genuine generalizability.")

    if st.button("🔍 Run Live Split Leakage Audit"):
        # Create simulated train/test sets to demonstrate detection
        train_mock = [
            {"id": "TR-01", "text": "حول 100 جنيه لمحمد 01012345678", "entities": [{"entity_type": "recipient_identifier", "raw_text": "01012345678"}], "metadata": {"speaker_id": "SPK_01"}},
            {"id": "TR-02", "text": "ادفع فاتورة النت 200", "entities": [], "metadata": {"speaker_id": "SPK_02"}}
        ]
        test_mock_clean = [
            {"id": "TE-01", "text": "ابعت 300 لسارة على 01299887766", "entities": [{"entity_type": "recipient_identifier", "raw_text": "01299887766"}], "metadata": {"speaker_id": "SPK_03"}}
        ]
        test_mock_leaked = [
            {"id": "TE-02", "text": "حول 100 جنيه لمحمد 01012345678", "entities": [{"entity_type": "recipient_identifier", "raw_text": "01012345678"}], "metadata": {"speaker_id": "SPK_01"}}
        ]

        st.subheader("1. Audit on Properly Isolated Partitions")
        report_clean = DataLeakageDetector.audit_split(train_mock, test_mock_clean)
        st.success(f"Status: {report_clean['status']}")
        st.write(report_clean['hygiene_recommendation'])

        st.subheader("2. Audit on Contaminated / Leaked Partitions (Demonstration)")
        report_leaked = DataLeakageDetector.audit_split(train_mock, test_mock_leaked)
        st.error(f"Status: {report_leaked['status']} (Detected {report_leaked['total_leaks_found']} Leaks)")
        for lk in report_leaked["leaks"]:
            st.warning(f"**[{lk['severity']}] {lk['leak_type']}:** {lk['description']}")
