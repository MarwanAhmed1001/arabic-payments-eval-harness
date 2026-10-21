# -*- coding: utf-8 -*-
"""
Baseline Rule-Based Runner.
Implements standard regular expressions and dictionary lookups.
Serves as the benchmark baseline against which all learned models are judged.
"""

import re
from typing import Dict, Any, List
from models.base_runner import BasePaymentModelRunner
from eval_harness.text_normalizer import normalize_arabic_orthography, normalize_digits, clean_egyptian_phone

class BaselineRuleRunner(BasePaymentModelRunner):
    """Regex & keyword lookup baseline model."""

    def __init__(self):
        super().__init__(model_name="Baseline-Rule-v1.0")

    def predict_intent(self, text: str) -> str:
        norm = normalize_arabic_orthography(text)

        # Naive keyword matching
        if any(w in norm for w in ["حول", "ابعت", "تحويل"]):
            return "P2P_TRANSFER"
        elif any(w in norm for w in ["ادفع", "سداد", "اشحن", "فاتوره", "فاتورة"]):
            return "BILL_PAYMENT"
        elif any(w in norm for w in ["رصيد", "استعلم", "متبقي", "فاضل"]):
            return "BALANCE_INQUIRY"
        else:
            return "OUT_OF_SCOPE"

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        norm_digits_text = normalize_digits(text)
        entities = []

        # Phone extraction: 11 digits starting with 01
        phone_match = re.search(r'\b(01[0125]\d{8})\b', norm_digits_text)
        phone = None
        if phone_match:
            phone = phone_match.group(1)
            entities.append({
                "entity_type": "recipient_identifier",
                "raw_text": phone,
                "normalized_value": phone
            })

        # Amount extraction: standalone number that is NOT the phone number
        for match in re.finditer(r'\b(\d+(?:\.\d+)?)\b', norm_digits_text):
            val_str = match.group(1)
            if phone and val_str in phone:
                continue
            if len(val_str) > 8:
                # likely an account number or bill code
                continue
            try:
                amt = float(val_str)
                entities.append({
                    "entity_type": "amount",
                    "raw_text": val_str,
                    "normalized_value": amt
                })
                break
            except ValueError:
                pass

        # Currency
        if re.search(r'\b(جنيه|جنية|ج\.م|ج)\b', text):
            entities.append({
                "entity_type": "currency",
                "raw_text": "جنيه",
                "normalized_value": "EGP"
            })

        # Payment rails & billers
        if "فودافون كاش" in text:
            entities.append({"entity_type": "payment_rail", "raw_text": "فودافون كاش", "normalized_value": "VODAFONE_CASH"})
        elif "انستاباي" in text or "انستا باي" in text:
            entities.append({"entity_type": "payment_rail", "raw_text": "انستاباي", "normalized_value": "INSTAPAY"})

        if "وي" in text or "we" in text.lower():
            entities.append({"entity_type": "biller_name", "raw_text": "وي", "normalized_value": "WE_TELECOM"})
        elif "الكهرباء" in text or "الكهربا" in text:
            entities.append({"entity_type": "biller_service", "raw_text": "الكهرباء", "normalized_value": "ELECTRICITY"})

        return entities
