# -*- coding: utf-8 -*-
"""
Experiment Comparison & Regression Tracker.
Performs diff analysis between two evaluation runs (e.g. Baseline vs Fine-Tuned/Prompt-Engineered),
highlighting fixed cases, regressions, and statistical metric shifts.
"""

from typing import Dict, Any, List

class RegressionTracker:
    """Detects regressions and performance deltas across experimental iterations."""

    @staticmethod
    def compare_runs(baseline_report: Dict[str, Any], experiment_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compares baseline metrics and error lists with experimental run.
        """
        b_intent = baseline_report.get("intent_metrics", {})
        e_intent = experiment_report.get("intent_metrics", {})

        b_entity = baseline_report.get("entity_metrics", {})
        e_entity = experiment_report.get("entity_metrics", {})

        # Deltas
        acc_delta = e_intent.get("accuracy", 0.0) - b_intent.get("accuracy", 0.0)
        intent_f1_delta = e_intent.get("f1_macro", 0.0) - b_intent.get("f1_macro", 0.0)
        entity_f1_delta = e_entity.get("micro_f1", 0.0) - b_entity.get("micro_f1", 0.0)
        ser_delta = e_entity.get("slot_error_rate", 0.0) - b_entity.get("slot_error_rate", 0.0)

        # Build error maps
        b_errors = {e["id"] for e in b_intent.get("errors", [])}
        e_errors = {e["id"] for e in e_intent.get("errors", [])}

        fixes = list(b_errors - e_errors)
        regressions = list(e_errors - b_errors)
        persistent_errors = list(b_errors & e_errors)

        return {
            "summary_deltas": {
                "intent_accuracy_delta": round(acc_delta, 4),
                "intent_macro_f1_delta": round(intent_f1_delta, 4),
                "entity_micro_f1_delta": round(entity_f1_delta, 4),
                "slot_error_rate_delta": round(ser_delta, 4),
            },
            "regression_analysis": {
                "fixed_cases_count": len(fixes),
                "fixed_sample_ids": fixes,
                "regression_cases_count": len(regressions),
                "regressed_sample_ids": regressions,
                "persistent_error_count": len(persistent_errors),
                "verdict": "RECOMMENDED" if acc_delta >= 0 and len(regressions) == 0 else "CAUTION_REGRESSIONS_DETECTED"
            }
        }
