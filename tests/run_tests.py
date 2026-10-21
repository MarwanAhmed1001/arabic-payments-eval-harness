# -*- coding: utf-8 -*-
"""
Standard Library Test Suite (unittest).
Executes without requiring external test dependencies.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from eval_harness.text_normalizer import normalize_arabic_orthography, clean_egyptian_phone, normalize_digits
from eval_harness.intent_evaluator import IntentEvaluator
from eval_harness.entity_evaluator import EntityEvaluator
from eval_harness.transcription_evaluator import TranscriptionEvaluator
from eval_harness.regression_tracker import RegressionTracker

class TestArabicEvalHarness(unittest.TestCase):

    def test_arabic_orthography_normalization(self):
        raw = "حَوِّلْ ٥٠٠ جُنَيْهٍ لِأَحْمَدَ على محفظة ٠١٠١٢٣٤٥٦٧٨"
        norm = normalize_arabic_orthography(raw)
        self.assertNotIn("٥", norm)
        self.assertIn("500", norm)
        self.assertIn("احمد", norm)
        self.assertIn("محفظه", norm)

    def test_egyptian_phone_cleaning(self):
        self.assertEqual(clean_egyptian_phone("+201012345678"), "01012345678")
        self.assertEqual(clean_egyptian_phone("00201012345678"), "01012345678")
        self.assertEqual(clean_egyptian_phone("010-1234-5678"), "01012345678")
        self.assertEqual(clean_egyptian_phone("0123456789"), "0123456789")

    def test_intent_evaluator_metrics(self):
        evaluator = IntentEvaluator()
        gold = [
            {"id": "1", "text": "حول 100", "intent": "P2P_TRANSFER"},
            {"id": "2", "text": "ادفع الفاتورة", "intent": "BILL_PAYMENT"},
            {"id": "3", "text": "نسيت الباسورد", "intent": "OUT_OF_SCOPE"}
        ]
        preds = [
            {"id": "1", "predicted_intent": "P2P_TRANSFER"},
            {"id": "2", "predicted_intent": "BILL_PAYMENT"},
            {"id": "3", "predicted_intent": "P2P_TRANSFER"}  # Critical False Accept!
        ]
        metrics = evaluator.evaluate(gold, preds)
        self.assertAlmostEqual(metrics["accuracy"], 2/3, places=2)
        self.assertEqual(metrics["fintech_safety"]["oos_false_accepts"], 1)
        self.assertEqual(metrics["fintech_safety"]["risk_status"], "CRITICAL")

    def test_entity_evaluator_value_matching(self):
        evaluator = EntityEvaluator()
        gold = [
            {
                "id": "1",
                "entities": [
                    {"entity_type": "amount", "normalized_value": 500.0},
                    {"entity_type": "recipient_identifier", "normalized_value": "01012345678"}
                ]
            }
        ]
        preds = [
            {
                "id": "1",
                "predicted_entities": [
                    {"entity_type": "amount", "normalized_value": 500},
                    {"entity_type": "recipient_identifier", "normalized_value": "+201012345678"}
                ]
            }
        ]
        metrics = evaluator.evaluate(gold, preds)
        self.assertEqual(metrics["turn_exact_match_rate"], 1.0)
        self.assertEqual(metrics["micro_f1"], 1.0)
        self.assertEqual(metrics["slot_error_rate"], 0.0)

    def test_transcription_evaluator_wer(self):
        evaluator = TranscriptionEvaluator()
        samples = [
            {
                "id": "1",
                "asr_reference": "حول خمسين جنيه لمحمد",
                "asr_hypothesis": "حول 50 جنيه لاحمد"
            }
        ]
        metrics = evaluator.evaluate(samples)
        self.assertEqual(metrics["total_audio_samples"], 1)
        self.assertGreater(metrics["raw_wer"], 0)
        self.assertLessEqual(metrics["normalized_wer"], metrics["raw_wer"])

if __name__ == "__main__":
    unittest.main()
