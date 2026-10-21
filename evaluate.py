# -*- coding: utf-8 -*-
"""
Master CLI Evaluation Harness for Arabic & Dialect Payment Models.
Runs standardized benchmarks on Intent, Entity, and Transcription tasks.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from tabulate import tabulate
from eval_harness.intent_evaluator import IntentEvaluator
from eval_harness.entity_evaluator import EntityEvaluator
from eval_harness.transcription_evaluator import TranscriptionEvaluator
from eval_harness.regression_tracker import RegressionTracker
from models.baseline_rule_runner import BaselineRuleRunner
from models.semantic_hybrid_runner import SemanticHybridRunner

def load_dataset(dataset_path: str) -> Dict[str, Any]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_evaluation(
    model_type: str,
    dataset_path: str,
    task: str = "all",
    output_dir: str = "reports/latest",
    compare_with: str = None
) -> Dict[str, Any]:
    print(f"\n================================================================================")
    print(f"  PARLEX ARABIC CONVERSATIONAL EVALUATION HARNESS")
    print(f"================================================================================")
    print(f"  Model Runner    : {model_type}")
    print(f"  Benchmark Suite : {dataset_path}")
    print(f"  Target Task     : {task}")
    print(f"================================================================================\n")

    # Load data
    data_payload = load_dataset(dataset_path)
    samples = data_payload.get("data", [])
    print(f"[*] Loaded {len(samples)} evaluation samples from benchmark v{data_payload.get('version', '1.0')}\n")

    # Select model runner
    if model_type == "baseline":
        runner = BaselineRuleRunner()
    elif model_type in ["semantic_hybrid", "hybrid"]:
        runner = SemanticHybridRunner()
    else:
        raise ValueError(f"Unknown model runner: {model_type}. Choose 'baseline' or 'semantic_hybrid'.")

    # Run predictions
    print(f"[*] Running inference across test set using [{runner.model_name}]...")
    predictions = runner.run_batch(samples)

    report: Dict[str, Any] = {
        "benchmark_metadata": {
            "version": data_payload.get("version"),
            "dataset_name": data_payload.get("benchmark_name"),
            "sample_count": len(samples)
        },
        "model_metadata": {
            "model_name": runner.model_name,
            "model_type": model_type
        }
    }

    # 1. Intent Classification Evaluation
    if task in ["all", "intent"]:
        print("\n[+] Evaluating Intent Classification Performance...")
        intent_eval = IntentEvaluator()
        intent_metrics = intent_eval.evaluate(samples, predictions)
        report["intent_metrics"] = intent_metrics

        intent_table = [
            ["Overall Accuracy", f"{intent_metrics['accuracy'] * 100:.2f}%"],
            ["Macro F1-Score", f"{intent_metrics['f1_macro']:.4f}"],
            ["Macro Precision", f"{intent_metrics['precision_macro']:.4f}"],
            ["Macro Recall", f"{intent_metrics['recall_macro']:.4f}"],
            ["Out-of-Scope False Accept Rate", f"{intent_metrics['fintech_safety']['oos_false_accept_rate'] * 100:.2f}% ({intent_metrics['fintech_safety']['risk_status']})"]
        ]
        print(tabulate(intent_table, headers=["Metric", "Result"], tablefmt="grid"))

        per_class_table = []
        for label, stats in intent_metrics["per_class"].items():
            per_class_table.append([label, stats["precision"], stats["recall"], stats["f1_score"], stats["support"]])
        print("\n[Intent Per-Class Performance]")
        print(tabulate(per_class_table, headers=["Class", "Precision", "Recall", "F1", "Support"], tablefmt="simple"))

    # 2. Entity Extraction Evaluation
    if task in ["all", "entity"]:
        print("\n[+] Evaluating Named Entity Recognition & Slot Extraction...")
        entity_eval = EntityEvaluator()
        entity_metrics = entity_eval.evaluate(samples, predictions)
        report["entity_metrics"] = entity_metrics

        entity_summary_table = [
            ["Turn Exact Match Rate (EM)", f"{entity_metrics['turn_exact_match_rate'] * 100:.2f}%"],
            ["Micro F1-Score", f"{entity_metrics['micro_f1']:.4f}"],
            ["Micro Precision", f"{entity_metrics['micro_precision']:.4f}"],
            ["Micro Recall", f"{entity_metrics['micro_recall']:.4f}"],
            ["Slot Error Rate (SER)", f"{entity_metrics['slot_error_rate']:.4f}"]
        ]
        print(tabulate(entity_summary_table, headers=["Entity Metric", "Score"], tablefmt="grid"))

        per_type_table = []
        for etype, stats in entity_metrics["per_entity_type"].items():
            if stats["support"] > 0 or stats["fp"] > 0:
                per_type_table.append([etype, stats["precision"], stats["recall"], stats["f1_score"], stats["support"]])
        print("\n[Per-Entity Breakdown]")
        print(tabulate(per_type_table, headers=["Slot Type", "Precision", "Recall", "F1", "Support"], tablefmt="simple"))

    # 3. Speech Transcription (ASR) Quality Evaluation
    if task in ["all", "transcription"]:
        print("\n[+] Evaluating Speech-to-Text (ASR) Quality on Dialect Audio...")
        trans_eval = TranscriptionEvaluator()
        trans_metrics = trans_eval.evaluate(samples)
        report["transcription_metrics"] = trans_metrics

        asr_table = [
            ["Raw Word Error Rate (WER)", f"{trans_metrics['raw_wer'] * 100:.2f}%"],
            ["Normalized Arabic WER", f"{trans_metrics['normalized_wer'] * 100:.2f}%"],
            ["Character Error Rate (CER)", f"{trans_metrics['raw_cer'] * 100:.2f}%"],
            ["Financial Keyword Error Rate (KER)", f"{trans_metrics['keyword_error_rate_ker'] * 100:.2f}%"],
            ["Orthography Impact Gain", f"-{trans_metrics['orthography_impact_gain'] * 100:.2f}% (Formatting Noise)"]
        ]
        print(tabulate(asr_table, headers=["ASR Metric", "Score"], tablefmt="grid"))

    # 4. Regression Analysis against a baseline
    if compare_with and os.path.exists(compare_with):
        print(f"\n[+] Computing Regression Analysis against Baseline [{compare_with}]...")
        with open(compare_with, "r", encoding="utf-8") as bf:
            baseline_rep = json.load(bf)
        reg_diff = RegressionTracker.compare_runs(baseline_rep, report)
        report["regression_comparison"] = reg_diff

        reg_table = [
            ["Intent Accuracy Shift", f"{reg_diff['summary_deltas']['intent_accuracy_delta'] * 100:+.2f}%"],
            ["Entity F1 Shift", f"{reg_diff['summary_deltas']['entity_micro_f1_delta']:+.4f}"],
            ["Fixed Test Cases (+)", reg_diff['regression_analysis']['fixed_cases_count']],
            ["Regressed Test Cases (-)", reg_diff['regression_analysis']['regression_cases_count']],
            ["Overall Status", reg_diff['regression_analysis']['verdict']]
        ]
        print(tabulate(reg_table, headers=["Regression Metric", "Delta"], tablefmt="grid"))

    # Export results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Export Findings Markdown
    md_path = out_path / "FINDINGS.md"
    generate_findings_markdown(report, md_path)

    print(f"\n[✓] Results successfully exported:")
    print(f"    - JSON Metrics  : {json_path}")
    print(f"    - Findings Docs : {md_path}\n")

    return report

def generate_findings_markdown(report: Dict[str, Any], output_md_path: Path):
    """Generates an executive findings document readable by engineering and product leads."""
    model_name = report["model_metadata"]["model_name"]
    intent = report.get("intent_metrics", {})
    entity = report.get("entity_metrics", {})
    trans = report.get("transcription_metrics", {})
    reg = report.get("regression_comparison", {})

    content = f"""# Executive Findings & Evaluation Report: {model_name}

**Date:** 2026-10-21  
**Benchmark Suite:** {report['benchmark_metadata']['dataset_name']} (v{report['benchmark_metadata']['version']})  
**Evaluator Author:** Marwan Ahmed (AI & Machine Learning)  

---

## 1. Executive Summary
This evaluation harness quantifies model performance across **Intent Classification**, **Financial Slot Extraction (NER)**, and **Egyptian Dialect ASR Transcription**. Rather than relying on eyeballed samples, this run delivers rigorous, reproducible metrics.

### Key Headline Metrics:
* **Intent Classification Accuracy:** `{intent.get('accuracy', 0.0) * 100:.2f}%` (Macro F1: `{intent.get('f1_macro', 0.0):.4f}`)
* **Entity Extraction Micro F1:** `{entity.get('micro_f1', 0.0):.4f}` (Exact Turn Match: `{entity.get('turn_exact_match_rate', 0.0) * 100:.2f}%`)
* **Slot Error Rate (SER):** `{entity.get('slot_error_rate', 0.0):.4f}`
* **Out-of-Scope (OOS) False Acceptance Rate:** `{intent.get('fintech_safety', {}).get('oos_false_accept_rate', 0.0) * 100:.2f}%` ({intent.get('fintech_safety', {}).get('risk_status', 'UNKNOWN')})
* **Normalized Egyptian ASR WER:** `{trans.get('normalized_wer', 0.0) * 100:.2f}%` (Raw WER: `{trans.get('raw_wer', 0.0) * 100:.2f}%`)

---

## 2. What the Numbers DO Tell You
1. **Financial Risk Mitigation:** The Out-of-Scope False Acceptance Rate is a critical metric for banking conversational agents. High general accuracy can mask dangerous failures where a complaint like *"الفيزا اتسحبت في الـ ATM"* or an explicit cancellation *"الغي التحويل"* is misclassified as an actionable transfer.
2. **Value-Normalized Entity Matching vs Raw String Matching:** In Egyptian dialect, amounts are frequently written as words (*"مية وخمسين"*, *"باكو"*) or Arabic-Indic numerals (*"٥٠٠"*). Strict token matching underestimates model utility; value-normalized matching confirms whether downstream banking APIs receive the exact canonical float/integer.
3. **ASR Orthography Impact:** Comparing Raw WER (`{trans.get('raw_wer', 0.0) * 100:.2f}%`) with Normalized Arabic WER (`{trans.get('normalized_wer', 0.0) * 100:.2f}%`) shows an **orthographic noise reduction of {trans.get('orthography_impact_gain', 0.0) * 100:.2f}%**, proving that surface letter mismatches (أ / ا / ة / ه) account for artificial acoustic penalties.

---

## 3. What the Numbers DO NOT Tell You (Edge Cases & Limitations)
* **Single-Turn vs Multi-Turn Ambiguity:** These benchmarks evaluate discrete turns. In real banking workflows, an omitted slot (e.g. missing recipient) should trigger a slot-request dialogue act rather than an outright classification error.
* **Dialect Slang Drift:** Terms like *"فكلي كارت شحن بعشرينيه"* require ongoing lexical calibration.
* **Audio Noise Profiles:** Clean simulated ASR does not capture background street noise or telephony codec compression (8kHz AMR/G.711).

---

## 4. Regression Analysis & Model Recommendations
"""
    if reg:
        deltas = reg.get("summary_deltas", {})
        analysis = reg.get("regression_analysis", {})
        content += f"""
* **Accuracy Shift:** `{deltas.get('intent_accuracy_delta', 0.0) * 100:+.2f}%`
* **Entity F1 Shift:** `{deltas.get('entity_micro_f1_delta', 0.0):+.4f}`
* **Fixed Cases:** `{analysis.get('fixed_cases_count', 0)}` test cases resolved.
* **Regressions:** `{analysis.get('regression_cases_count', 0)}` previous capabilities broken.
* **Deployment Verdict:** `{analysis.get('verdict')}`
"""
    else:
        content += "\n*Baseline run recorded. Subsequent runs with `--compare-with` will compute automated diffs and regressions.*\n"

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(content)

def main():
    parser = argparse.ArgumentParser(description="Parlex Arabic Payment Evaluation Harness CLI")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "semantic_hybrid", "hybrid"],
                        help="Model runner to evaluate")
    parser.add_argument("--dataset", type=str, default="data/v1.0_golden_eval.json",
                        help="Path to evaluation dataset JSON")
    parser.add_argument("--task", type=str, default="all", choices=["all", "intent", "entity", "transcription"],
                        help="Specific task to benchmark")
    parser.add_argument("--output-dir", type=str, default="reports/latest",
                        help="Directory to save output reports")
    parser.add_argument("--compare-with", type=str, default=None,
                        help="Path to prior evaluation report JSON for regression analysis")

    args = parser.parse_args()
    run_evaluation(
        model_type=args.model,
        dataset_path=args.dataset,
        task=args.task,
        output_dir=args.output_dir,
        compare_with=args.compare_with
    )

if __name__ == "__main__":
    main()
