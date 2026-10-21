# -*- coding: utf-8 -*-
"""
Unit Tests for Arabic Payments Evaluation Harness.
Verifies normalization, intent evaluation, entity matching, and ASR WER calculations.
"""

import pytest
from eval_harness.text_normalizer import normalize_arabic_orthography, clean_egyptian_phone, normalize_digits
from eval_harness.intent_evaluator import IntentEvaluator
from eval_harness.entity_evaluator import EntityEvaluator
from eval_harness.transcription_evaluator import TranscriptionEvaluator
from eval_harness.regression_tracker import RegressionTracker

def test_arabic_orthography_normalization():
    # Test Alef, Taa Marbouta, Tashkeel, and Eastern Arabic digits
    raw = "حَوِّلْ ٥٠٠ جُنَيْهٍ لِأَحْمَدَ على محفظة ٠١٠١٢٣٤٥٦٧٨"
    norm = normalize_arabic_orthography(raw)
    assert "٥" not in norm
    assert "500" in norm
    assert "احمد" in norm
    assert "محفظه" in norm

def test_egyptian_phone_cleaning():
    assert clean_egyptian_phone("+201012345678") == "01012345678"
    assert clean_egyptian_phone("00201012345678") == "01012345678"
    assert clean_egyptian_phone("010-1234-5678") == "01012345678"
    assert clean_egyptian_phone("0123456789") == "0123456789"

def test_intent_evaluator_metrics():
    evaluator = IntentEvaluator()
    gold = [
        {"id": "1", "text": "حول 100", "intent": "P2P_TRANSFER"},
        {"id": "2", "text": "ادفع الفاتورة", "intent": "BILL_PAYMENT"},
        {"id": "3", "text": "نسيت الباسورد", "intent": "OUT_OF_SCOPE"}
    ]
    preds = [
        {"id": "1", "predicted_intent": "P2P_TRANSFER"},
        {"id": "2", "predicted_intent": "BILL_PAYMENT"},
        {"id": "3", "predicted_intent": "P2P_TRANSFER"} # Dangerous False Accept!
    ]
    metrics = evaluator.evaluate(gold, preds)
    assert metrics["accuracy"] == pytest.approx(2/3, 0.01)
    assert metrics["fintech_safety"]["oos_false_accepts"] == 1
    assert metrics["fintech_safety"]["risk_status"] == "CRITICAL"

def test_entity_evaluator_value_matching():
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
    assert metrics["turn_exact_match_rate"] == 1.0
    assert metrics["micro_f1"] == 1.0
    assert metrics["slot_error_rate"] == 0.0

def test_transcription_evaluator_wer():
    evaluator = TranscriptionEvaluator()
    samples = [
        {
            "id": "1",
            "asr_reference": "حول خمسين جنيه لمحمد",
            "asr_hypothesis": "حول 50 جنيه لاحمد"
        }
    ]
    metrics = evaluator.evaluate(samples)
    assert metrics["total_audio_samples"] == 1
    assert metrics["raw_wer"] > 0
    # Normalized WER should be equal or lower than raw WER
    assert metrics["normalized_wer"] <= metrics["raw_wer"]
