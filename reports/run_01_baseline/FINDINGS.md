# Executive Findings & Evaluation Report: Baseline-Rule-v1.0

**Date:** 2026-10-21  
**Benchmark Suite:** Parlex Arabic & Egyptian Dialect Payment Evaluation Suite (v1.0.0)  
**Evaluator Author:** Marwan Ahmed (AI & Machine Learning)  

---

## 1. Executive Summary
This evaluation harness quantifies model performance across **Intent Classification**, **Financial Slot Extraction (NER)**, and **Egyptian Dialect ASR Transcription**. Rather than relying on eyeballed samples, this run delivers rigorous, reproducible metrics.

### Key Headline Metrics:
* **Intent Classification Accuracy:** `100.00%` (Macro F1: `1.0000`)
* **Entity Extraction Micro F1:** `0.6479` (Exact Turn Match: `31.25%`)
* **Slot Error Rate (SER):** `0.5435`
* **Out-of-Scope (OOS) False Acceptance Rate:** `0.00%` (PASS)
* **Normalized Egyptian ASR WER:** `47.33%` (Raw WER: `61.73%`)

---

## 2. What the Numbers DO Tell You
1. **Financial Risk Mitigation:** The Out-of-Scope False Acceptance Rate is a critical metric for banking conversational agents. High general accuracy can mask dangerous failures where a complaint like *"الفيزا اتسحبت في الـ ATM"* or an explicit cancellation *"الغي التحويل"* is misclassified as an actionable transfer.
2. **Value-Normalized Entity Matching vs Raw String Matching:** In Egyptian dialect, amounts are frequently written as words (*"مية وخمسين"*, *"باكو"*) or Arabic-Indic numerals (*"٥٠٠"*). Strict token matching underestimates model utility; value-normalized matching confirms whether downstream banking APIs receive the exact canonical float/integer.
3. **ASR Orthography Impact:** Comparing Raw WER (`61.73%`) with Normalized Arabic WER (`47.33%`) shows an **orthographic noise reduction of 14.40%**, proving that surface letter mismatches (أ / ا / ة / ه) account for artificial acoustic penalties.

---

## 3. What the Numbers DO NOT Tell You (Edge Cases & Limitations)
* **Single-Turn vs Multi-Turn Ambiguity:** These benchmarks evaluate discrete turns. In real banking workflows, an omitted slot (e.g. missing recipient) should trigger a slot-request dialogue act rather than an outright classification error.
* **Dialect Slang Drift:** Terms like *"فكلي كارت شحن بعشرينيه"* require ongoing lexical calibration.
* **Audio Noise Profiles:** Clean simulated ASR does not capture background street noise or telephony codec compression (8kHz AMR/G.711).

---

## 4. Regression Analysis & Model Recommendations

*Baseline run recorded. Subsequent runs with `--compare-with` will compute automated diffs and regressions.*
