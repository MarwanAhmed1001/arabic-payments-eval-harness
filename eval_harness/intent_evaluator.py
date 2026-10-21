# -*- coding: utf-8 -*-
"""
Intent Classification Evaluator.
Computes standard classification metrics, per-class performance, confusion matrices,
and banking safety metrics (e.g. False Acceptance of Out-of-Scope instructions).
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class IntentEvaluator:
    """Evaluates conversational payment intent classification systems."""

    def __init__(self, labels: Optional[List[str]] = None):
        self.labels = labels or [
            "P2P_TRANSFER",
            "BILL_PAYMENT",
            "BALANCE_INQUIRY",
            "OUT_OF_SCOPE"
        ]

    def evaluate(self, gold_samples: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates metrics between ground truth and predicted intents.
        
        Args:
            gold_samples: List of reference dicts containing 'id', 'text', 'intent'
            predictions: List of prediction dicts containing 'id', 'predicted_intent'
        """
        pred_map = {p["id"]: p.get("predicted_intent", "UNKNOWN") for p in predictions}

        y_true = []
        y_pred = []
        errors = []

        for sample in gold_samples:
            sample_id = sample["id"]
            true_intent = sample["intent"]
            pred_intent = pred_map.get(sample_id, "UNKNOWN")

            y_true.append(true_intent)
            y_pred.append(pred_intent)

            if true_intent != pred_intent:
                errors.append({
                    "id": sample_id,
                    "text": sample["text"],
                    "true_intent": true_intent,
                    "predicted_intent": pred_intent
                })

        # Base accuracy
        accuracy = float(accuracy_score(y_true, y_pred))

        # Precision, Recall, F1 (macro and weighted)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=self.labels, average="macro", zero_division=0
        )
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=self.labels, average="weighted", zero_division=0
        )

        # Per-class metrics
        p_per, r_per, f1_per, support_per = precision_recall_fscore_support(
            y_true, y_pred, labels=self.labels, average=None, zero_division=0
        )

        per_class_metrics = {}
        for idx, label in enumerate(self.labels):
            per_class_metrics[label] = {
                "precision": round(float(p_per[idx]), 4),
                "recall": round(float(r_per[idx]), 4),
                "f1_score": round(float(f1_per[idx]), 4),
                "support": int(support_per[idx])
            }

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=self.labels)

        # FinTech Safety Metric: False Acceptance of Out-of-Scope (OOS) queries
        # If true is OUT_OF_SCOPE, but predicted is an actionable financial intent (Transfer or Bill),
        # this represents a severe risk of unauthorized or accidental debit.
        oos_total = 0
        oos_false_accepts = 0
        for sample in gold_samples:
            if sample["intent"] == "OUT_OF_SCOPE":
                oos_total += 1
                pred = pred_map.get(sample["id"], "UNKNOWN")
                if pred in ["P2P_TRANSFER", "BILL_PAYMENT"]:
                    oos_false_accepts += 1

        oos_false_accept_rate = float(oos_false_accepts / oos_total) if oos_total > 0 else 0.0

        return {
            "total_samples": len(gold_samples),
            "accuracy": round(accuracy, 4),
            "f1_macro": round(float(f1_macro), 4),
            "precision_macro": round(float(p_macro), 4),
            "recall_macro": round(float(r_macro), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "per_class": per_class_metrics,
            "confusion_matrix": cm.tolist(),
            "labels": self.labels,
            "fintech_safety": {
                "oos_total": oos_total,
                "oos_false_accepts": oos_false_accepts,
                "oos_false_accept_rate": round(oos_false_accept_rate, 4),
                "risk_status": "CRITICAL" if oos_false_accept_rate > 0.1 else "PASS"
            },
            "error_count": len(errors),
            "errors": errors
        }
