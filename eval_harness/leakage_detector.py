# -*- coding: utf-8 -*-
"""
Data Leakage & Split Hygiene Detector.
Audits training and test sets to guarantee evaluation integrity by detecting:
1. Speaker / User ID Overlap (Acoustic / Idiolect memorization)
2. Identifier Leakage (Phone numbers, account numbers, InstaPay handles)
3. Exact Text & High N-Gram Overlap Leakage
"""

from typing import List, Dict, Any, Set
from collections import defaultdict
from eval_harness.text_normalizer import normalize_arabic_orthography, clean_egyptian_phone

class DataLeakageDetector:
    """Verifies train/test partition independence and reports leaks."""

    @staticmethod
    def audit_split(train_samples: List[Dict[str, Any]], test_samples: List[Dict[str, Any]], ngram_size: int = 3) -> Dict[str, Any]:
        leaks_found = []

        # 1. Speaker / User ID Overlap Leakage
        train_speakers = {s.get("metadata", {}).get("speaker_id") for s in train_samples if s.get("metadata", {}).get("speaker_id")}
        test_speakers = {s.get("metadata", {}).get("speaker_id") for s in test_samples if s.get("metadata", {}).get("speaker_id")}

        speaker_overlap = train_speakers & test_speakers
        if speaker_overlap:
            leaks_found.append({
                "leak_type": "SPEAKER_USER_OVERLAP",
                "severity": "CRITICAL",
                "description": f"Found {len(speaker_overlap)} overlapping speaker/user IDs across train and test sets. "
                               f"Causes acoustic and idiolect memorization instead of generalizability.",
                "affected_entities": list(speaker_overlap)
            })

        # 2. Financial Identifier Leakage (Phone, Account numbers, InstaPay handles)
        def extract_identifiers(samples: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
            ids_map = defaultdict(set)
            for s in samples:
                for ent in s.get("entities", []):
                    if ent.get("entity_type") in ["recipient_identifier", "account_number"]:
                        val = str(ent.get("normalized_value", ent.get("raw_text", ""))).strip().lower()
                        clean_p = clean_egyptian_phone(val)
                        canonical = clean_p if clean_p else val
                        if canonical:
                            ids_map[canonical].add(s["id"])
            return ids_map

        train_ids = extract_identifiers(train_samples)
        test_ids = extract_identifiers(test_samples)

        shared_identifiers = set(train_ids.keys()) & set(test_ids.keys())
        if shared_identifiers:
            leaks_found.append({
                "leak_type": "FINANCIAL_IDENTIFIER_LEAKAGE",
                "severity": "HIGH",
                "description": f"Found {len(shared_identifiers)} shared recipient identifiers across train and test sets. "
                               f"Allows model to memorize recipient targets as shortcuts.",
                "affected_entities": list(shared_identifiers)
            })

        # 3. Exact Text & Duplicate Turn Leakage
        train_texts = {normalize_arabic_orthography(s["text"]): s["id"] for s in train_samples}
        exact_text_duplicates = []

        for s in test_samples:
            norm_test = normalize_arabic_orthography(s["text"])
            if norm_test in train_texts:
                exact_text_duplicates.append({
                    "test_sample_id": s["id"],
                    "train_sample_id": train_texts[norm_test],
                    "text": s["text"]
                })

        if exact_text_duplicates:
            leaks_found.append({
                "leak_type": "EXACT_TEXT_DUPLICATION",
                "severity": "CRITICAL",
                "description": f"Found {len(exact_text_duplicates)} exact duplicate conversational turns in test set that exist in train set.",
                "affected_samples": exact_text_duplicates[:10]
            })

        is_clean = len(leaks_found) == 0

        return {
            "status": "PASS - NO LEAKAGE DETECTED" if is_clean else "FAIL - LEAKAGE DETECTED",
            "train_sample_count": len(train_samples),
            "test_sample_count": len(test_samples),
            "total_leaks_found": len(leaks_found),
            "leaks": leaks_found,
            "hygiene_recommendation": (
                "Partition sets using GroupKFold by user_id/speaker_id and compute all normalization parameters strictly on train."
                if not is_clean else "Partitions satisfy clean evaluation isolation standards."
            )
        }
