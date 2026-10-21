# -*- coding: utf-8 -*-
"""
Entity Extraction Evaluator for Arabic Payment Instructions.
Evaluates Named Entity Recognition (NER) & slot extraction with both strict
exact-matching and normalized semantic matching (e.g. phone numbers, amounts).
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
from rapidfuzz import fuzz
from eval_harness.text_normalizer import normalize_arabic_orthography, clean_egyptian_phone, normalize_digits

class EntityEvaluator:
    """Evaluates slot filling & entity extraction in financial conversations."""

    ENTITY_TYPES = [
        "amount",
        "currency",
        "recipient_name",
        "recipient_identifier",
        "biller_name",
        "biller_service",
        "payment_rail"
    ]

    def __init__(self, fuzzy_threshold: float = 85.0):
        self.fuzzy_threshold = fuzzy_threshold

    def _values_match(self, entity_type: str, true_val: Any, pred_val: Any) -> bool:
        """Determines whether true entity value and predicted entity value match."""
        if true_val is None or pred_val is None:
            return False

        if entity_type == "amount":
            try:
                t_float = float(normalize_digits(str(true_val)))
                p_float = float(normalize_digits(str(pred_val)))
                return abs(t_float - p_float) < 1e-4
            except (ValueError, TypeError):
                pass

        if entity_type == "recipient_identifier":
            # Check phone normalization
            t_clean = clean_egyptian_phone(str(true_val))
            p_clean = clean_egyptian_phone(str(pred_val))
            if t_clean and p_clean and t_clean == p_clean:
                return True
            # InstaPay handle check
            t_str = str(true_val).strip().lower()
            p_str = str(pred_val).strip().lower()
            return t_str == p_str

        # Standard text matching with Arabic orthography normalization
        norm_true = normalize_arabic_orthography(str(true_val))
        norm_pred = normalize_arabic_orthography(str(pred_val))

        if norm_true == norm_pred:
            return True

        # Fuzzy string similarity for recipient names
        if entity_type in ["recipient_name", "biller_name"]:
            similarity = fuzz.ratio(norm_true, norm_pred)
            return similarity >= self.fuzzy_threshold

        return False

    def evaluate(self, gold_samples: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates precision, recall, F1, exact match, and slot error rate (SER).
        """
        pred_map = {p["id"]: p.get("predicted_entities", []) for p in predictions}

        type_stats = {t: {"tp": 0, "fp": 0, "fn": 0} for t in self.ENTITY_TYPES}
        turn_exact_matches = 0
        total_turns = len(gold_samples)
        mismatches = []

        total_gold_slots = 0
        total_pred_slots = 0
        total_substitutions = 0
        total_deletions = 0
        total_insertions = 0

        for sample in gold_samples:
            sample_id = sample["id"]
            gold_ents = sample.get("entities", [])
            pred_ents = pred_map.get(sample_id, [])

            total_gold_slots += len(gold_ents)
            total_pred_slots += len(pred_ents)

            # Map predicted entities by type
            matched_pred_indices = set()
            turn_has_error = False

            for g_ent in gold_ents:
                etype = g_ent["entity_type"]
                g_val = g_ent.get("normalized_value", g_ent.get("raw_text"))

                found_match = False
                for p_idx, p_ent in enumerate(pred_ents):
                    if p_idx in matched_pred_indices:
                        continue
                    if p_ent.get("entity_type") == etype:
                        p_val = p_ent.get("normalized_value", p_ent.get("raw_text"))
                        if self._values_match(etype, g_val, p_val):
                            found_match = True
                            matched_pred_indices.add(p_idx)
                            if etype in type_stats:
                                type_stats[etype]["tp"] += 1
                            break

                if not found_match:
                    turn_has_error = True
                    total_deletions += 1
                    if etype in type_stats:
                        type_stats[etype]["fn"] += 1
                    mismatches.append({
                        "sample_id": sample_id,
                        "error_type": "MISSING_ENTITY (FN)",
                        "entity_type": etype,
                        "expected": g_val
                    })

            # Any unmatched predicted entities are False Positives / Insertions
            for p_idx, p_ent in enumerate(pred_ents):
                if p_idx not in matched_pred_indices:
                    turn_has_error = True
                    total_insertions += 1
                    etype = p_ent.get("entity_type", "UNKNOWN")
                    if etype in type_stats:
                        type_stats[etype]["fp"] += 1
                    mismatches.append({
                        "sample_id": sample_id,
                        "error_type": "SPURIOUS_ENTITY (FP)",
                        "entity_type": etype,
                        "predicted": p_ent.get("normalized_value", p_ent.get("raw_text"))
                    })

            if not turn_has_error:
                turn_exact_matches += 1

        # Calculate per-type Precision, Recall, F1
        per_type_metrics = {}
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for etype, stats in type_stats.items():
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]
            total_tp += tp
            total_fp += fp
            total_fn += fn

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            per_type_metrics[etype] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support": tp + fn
            }

        # Overall Micro metrics
        micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

        # Exact Match (EM) Turn Rate
        turn_em_rate = turn_exact_matches / total_turns if total_turns > 0 else 0.0

        # Slot Error Rate (SER) = (Substitutions + Deletions + Insertions) / Total Gold Slots
        ser = (total_substitutions + total_deletions + total_insertions) / total_gold_slots if total_gold_slots > 0 else 0.0

        return {
            "total_samples": total_turns,
            "turn_exact_match_rate": round(turn_em_rate, 4),
            "micro_precision": round(micro_prec, 4),
            "micro_recall": round(micro_rec, 4),
            "micro_f1": round(micro_f1, 4),
            "slot_error_rate": round(ser, 4),
            "per_entity_type": per_type_metrics,
            "mismatch_count": len(mismatches),
            "sample_mismatches": mismatches[:15]
        }
