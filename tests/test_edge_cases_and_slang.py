# -*- coding: utf-8 -*-
"""
Tests for Egyptian Dialect Slang, Negations, and Edge Cases.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from models.semantic_hybrid_runner import SemanticHybridRunner

class TestDialectSlangAndEdgeCases(unittest.TestCase):

    def setUp(self):
        self.runner = SemanticHybridRunner()

    def test_egyptian_slang_numerals(self):
        # 1. باكو = 1000 EGP
        text_bako = "ابعت باكو لمصطفى على المحفظة 01555667788"
        ents = self.runner.extract_entities(text_bako)
        amt_ent = next((e for e in ents if e["entity_type"] == "amount"), None)
        self.assertIsNotNone(amt_ent)
        self.assertEqual(amt_ent["normalized_value"], 1000.0)

        # 2. عشرينيه = 20 EGP
        text_eshrenya = "فكلي كارت شحن بعشرينيه لرقمي 01098765432"
        ents = self.runner.extract_entities(text_eshrenya)
        amt_ent = next((e for e in ents if e["entity_type"] == "amount"), None)
        self.assertIsNotNone(amt_ent)
        self.assertEqual(amt_ent["normalized_value"], 20.0)

    def test_instapay_handle_extraction(self):
        text = "ابعت 200 جنيه لـ سارة محمود على انستاباي sara@instapay"
        ents = self.runner.extract_entities(text)
        id_ent = next((e for e in ents if e["entity_type"] == "recipient_identifier"), None)
        self.assertIsNotNone(id_ent)
        self.assertEqual(id_ent["normalized_value"], "sara@instapay")

    def test_number_disambiguation_phone_vs_amount(self):
        text = "حول 100 جنيه لرقم 01011112222"
        ents = self.runner.extract_entities(text)
        amt_ent = next((e for e in ents if e["entity_type"] == "amount"), None)
        phone_ent = next((e for e in ents if e["entity_type"] == "recipient_identifier"), None)

        self.assertIsNotNone(amt_ent)
        self.assertEqual(amt_ent["normalized_value"], 100.0)
        self.assertIsNotNone(phone_ent)
        self.assertEqual(phone_ent["normalized_value"], "01011112222")

    def test_negations_and_cancellations(self):
        # Must be OUT_OF_SCOPE and yield 0 entities to prevent unauthorized transaction
        text = "لا خلاص الغي التحويل بتاع الـ 500 جنيه لمحمد اوعى تبعت حاجة"
        intent = self.runner.predict_intent(text)
        ents = self.runner.extract_entities(text)

        self.assertEqual(intent, "OUT_OF_SCOPE")
        self.assertEqual(len(ents), 0)

    def test_false_friend_numbers_in_complaints(self):
        # 500 here is "500 times", NOT 500 EGP payment!
        text = "انا كلمت خدمة العملاء 500 مرة ومحدش رد عليا"
        intent = self.runner.predict_intent(text)
        ents = self.runner.extract_entities(text)

        self.assertEqual(intent, "OUT_OF_SCOPE")
        amt_ent = next((e for e in ents if e["entity_type"] == "amount"), None)
        self.assertIsNone(amt_ent)

if __name__ == "__main__":
    unittest.main()
