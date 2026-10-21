# -*- coding: utf-8 -*-
"""
Tests for Data Leakage & Split Hygiene Detection.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from eval_harness.leakage_detector import DataLeakageDetector

class TestDataLeakageDetector(unittest.TestCase):

    def setUp(self):
        self.train_samples = [
            {
                "id": "TR-01",
                "text": "حول 500 جنيه لمحمد على فودافون كاش",
                "entities": [
                    {"entity_type": "recipient_identifier", "raw_text": "01012345678"}
                ],
                "metadata": {"speaker_id": "SPK_01"}
            },
            {
                "id": "TR-02",
                "text": "ادفع فاتورة الكهرباء جنوب القاهرة",
                "entities": [
                    {"entity_type": "recipient_identifier", "raw_text": "99887766"}
                ],
                "metadata": {"speaker_id": "SPK_02"}
            }
        ]

    def test_clean_split_no_leakage(self):
        clean_test = [
            {
                "id": "TE-01",
                "text": "ابعت ميتين جنيه لسارة على انستاباي",
                "entities": [
                    {"entity_type": "recipient_identifier", "raw_text": "01299887766"}
                ],
                "metadata": {"speaker_id": "SPK_03"}
            }
        ]
        audit = DataLeakageDetector.audit_split(self.train_samples, clean_test)
        self.assertEqual(audit["status"], "PASS - NO LEAKAGE DETECTED")
        self.assertEqual(audit["total_leaks_found"], 0)

    def test_detect_speaker_leakage(self):
        leaked_speaker_test = [
            {
                "id": "TE-02",
                "text": "عايز اعرف رصيدي كام",
                "entities": [],
                "metadata": {"speaker_id": "SPK_01"} # Overlaps with TR-01!
            }
        ]
        audit = DataLeakageDetector.audit_split(self.train_samples, leaked_speaker_test)
        self.assertEqual(audit["status"], "FAIL - LEAKAGE DETECTED")
        self.assertTrue(any(l["leak_type"] == "SPEAKER_USER_OVERLAP" for l in audit["leaks"]))

    def test_detect_financial_identifier_leakage(self):
        leaked_phone_test = [
            {
                "id": "TE-03",
                "text": "ابعت 20 جنيه للرقم ده 01012345678",
                "entities": [
                    {"entity_type": "recipient_identifier", "raw_text": "+201012345678"} # Normalizes to 01012345678
                ],
                "metadata": {"speaker_id": "SPK_04"}
            }
        ]
        audit = DataLeakageDetector.audit_split(self.train_samples, leaked_phone_test)
        self.assertEqual(audit["status"], "FAIL - LEAKAGE DETECTED")
        self.assertTrue(any(l["leak_type"] == "FINANCIAL_IDENTIFIER_LEAKAGE" for l in audit["leaks"]))

if __name__ == "__main__":
    unittest.main()
