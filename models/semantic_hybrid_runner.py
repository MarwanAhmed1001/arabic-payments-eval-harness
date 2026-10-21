# -*- coding: utf-8 -*-
"""
Semantic Hybrid Runner (v2.0).
Combines contextual rules, negation detection, Egyptian dialect number resolution,
and entity boundaries. Solves edge cases where baseline rule models fail.
"""

import re
from typing import Dict, Any, List
from models.base_runner import BasePaymentModelRunner
from eval_harness.text_normalizer import normalize_arabic_orthography, normalize_digits, clean_egyptian_phone

# Egyptian & Arabic spoken numbers vocabulary
COLLOQUIAL_NUMERALS = {
    "مية": 100.0,
    "ميه": 100.0,
    "ميتين": 200.0,
    "مائتين": 200.0,
    "مية وخمسين": 150.0,
    "ميه وخمسين": 150.0,
    "خمسمية": 500.0,
    "خمسمائة": 500.0,
    "سبعمية": 700.0,
    "سبعمائة": 700.0,
    "سبعمية وخمسين": 750.0,
    "باكو": 1000.0,
    "ألف": 1000.0,
    "الف": 1000.0,
    "ألفين": 2000.0,
    "الفين": 2000.0,
    "تلات آلاف": 3000.0,
    "تلات تلاف": 3000.0,
    "عشرينيه": 20.0,
    "عشرينية": 20.0,
    "خمسين": 50.0,
}

NEGATION_PATTERNS = [
    r'لا\s+خلاص',
    r'الغي\s+(التحويل|العملية)',
    r'اوعى\s+(تدفع|تبعت|تحول)',
    r'متحولش',
    r'متبعتش'
]

COMPLAINT_PATTERNS = [
    r'\d+\s+مرة',
    r'خدمة\s+العملاء',
    r'الفيزا\s+اتسحبت',
    r'كارتي\s+اتسحب',
    r'غير\s+كلمة\s+السر',
    r'نسيت\s+الباسورد',
    r'مواعيد\s+الشحن',
    r'عروض\s+باقات',
    r'الراوتر\s+.*فاصل'
]

class SemanticHybridRunner(BasePaymentModelRunner):
    """Contextual hybrid model with negation handling and dialect numerals."""

    def __init__(self):
        super().__init__(model_name="Semantic-Hybrid-Dialect-v2.0")

    def predict_intent(self, text: str) -> str:
        norm = normalize_arabic_orthography(text)

        # 1. Check for explicit cancellations or negations (Must route to OUT_OF_SCOPE / SAFE_CANCEL)
        for neg in NEGATION_PATTERNS:
            if re.search(neg, text) or re.search(neg, norm):
                return "OUT_OF_SCOPE"

        # 2. Check for customer support complaints or non-transactional queries
        for comp in COMPLAINT_PATTERNS:
            if re.search(comp, text) or re.search(comp, norm):
                return "OUT_OF_SCOPE"

        # 3. Balance inquiries
        if any(w in norm for w in ["رصيدي", "رصيد", "استعلم", "متبقي", "فاضل"]):
            if not any(w in norm for w in ["حول", "ابعت", "ادفع", "سداد"]):
                return "BALANCE_INQUIRY"

        # 4. Bill payments & Recharges
        if any(w in norm for w in ["ادفع", "سداد", "اسدد", "اشحن", "فاتورة", "فاتوره", "كارت شحن"]):
            return "BILL_PAYMENT"

        # 5. P2P Transfers
        if any(w in norm for w in ["حول", "ابعت", "احول", "تحويل"]):
            return "P2P_TRANSFER"

        return "OUT_OF_SCOPE"

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        norm_digits = normalize_digits(text)
        entities = []

        # 1. Negation check: if cancelled, do not emit executable entities
        for neg in NEGATION_PATTERNS:
            if re.search(neg, text):
                return []

        # False friends check: ignore number if it is "500 مرة" or "3 ايام"
        clean_text_for_amt = norm_digits
        clean_text_for_amt = re.sub(r'\d+\s+(مرة|ايام|يوم|ساعة)', '', clean_text_for_amt)

        # 2. InstaPay handle extraction (e.g. sara@instapay)
        instapay_match = re.search(r'([a-zA-Z0-9_\.]+@instapay)', text, re.IGNORECASE)
        instapay_handle = None
        if instapay_match:
            instapay_handle = instapay_match.group(1).lower()
            entities.append({
                "entity_type": "recipient_identifier",
                "raw_text": instapay_handle,
                "normalized_value": instapay_handle
            })

        # 3. Egyptian Mobile number extraction (010, 011, 012, 015)
        phone_match = re.search(r'\b(01[0125]\d{8})\b', norm_digits)
        phone = None
        if phone_match:
            phone = phone_match.group(1)
            entities.append({
                "entity_type": "recipient_identifier",
                "raw_text": phone,
                "normalized_value": phone
            })

        # Landline / Bill code extraction
        if not phone and not instapay_handle:
            bill_code_match = re.search(r'\b(02\d{8}|\d{6,14})\b', norm_digits)
            if bill_code_match:
                bcode = bill_code_match.group(1)
                entities.append({
                    "entity_type": "recipient_identifier",
                    "raw_text": bcode,
                    "normalized_value": bcode
                })

        # 4. Amount extraction: Check colloquial numerals first
        found_amount = False
        for phrase, val in sorted(COLLOQUIAL_NUMERALS.items(), key=lambda x: -len(x[0])):
            if phrase in text or phrase in clean_text_for_amt:
                entities.append({
                    "entity_type": "amount",
                    "raw_text": phrase,
                    "normalized_value": val
                })
                found_amount = True
                break

        # Digits fallback for amount
        if not found_amount:
            for match in re.finditer(r'\b(\d+(?:\.\d+)?)\b', clean_text_for_amt):
                val_str = match.group(1)
                if phone and val_str in phone:
                    continue
                if len(val_str) > 7:
                    continue
                try:
                    amt = float(val_str)
                    entities.append({
                        "entity_type": "amount",
                        "raw_text": val_str,
                        "normalized_value": amt
                    })
                    found_amount = True
                    break
                except ValueError:
                    pass

        # 5. Currency
        if re.search(r'\b(جنيه|جنية|ج\.م|ج|جنيه مصري)\b', text):
            entities.append({
                "entity_type": "currency",
                "raw_text": "جنيه",
                "normalized_value": "EGP"
            })

        # 6. Recipient Name Extraction (Pattern matching around prepositions: لـ, لمحمد, باسم)
        name_match = re.search(r'(?:لـ\s*|لـ|ل|باسم\s+)([أ-ي\s]{3,20})(?=\s+(?:على|علي|محفظة|رقم|حساب|بمبلغ|01|\d)|$)', text)
        if name_match:
            raw_name = name_match.group(1).strip()
            # Clean common preposition prefixes
            raw_name = re.sub(r'^(ـ|ـ\s*)', '', raw_name)
            if len(raw_name.split()) <= 3 and not any(w in raw_name for w in ["فودافون", "اورنج", "اتصالات", "وي", "البنك", "المحفظة"]):
                entities.append({
                    "entity_type": "recipient_name",
                    "raw_text": raw_name,
                    "normalized_value": raw_name
                })

        # 7. Payment Rail & Billers
        if "فودافون كاش" in text:
            entities.append({"entity_type": "payment_rail", "raw_text": "فودافون كاش", "normalized_value": "VODAFONE_CASH"})
        elif "انستاباي" in text or "انستا" in text:
            entities.append({"entity_type": "payment_rail", "raw_text": "انستاباي", "normalized_value": "INSTAPAY"})
        elif "اورنج" in text and "كاش" in text or "محفظة اورنج" in text:
            entities.append({"entity_type": "payment_rail", "raw_text": "اورنج", "normalized_value": "ORANGE_CASH"})
        elif "البنك الاهلي" in text or "البنك الأهلي" in text:
            entities.append({"entity_type": "payment_rail", "raw_text": "البنك الاهلي", "normalized_value": "NBE_BANK"})
        elif "ميزة" in text:
            entities.append({"entity_type": "payment_rail", "raw_text": "ميزة", "normalized_value": "MEEZA"})

        if "وي" in text or "شركة وي" in text:
            entities.append({"entity_type": "biller_name", "raw_text": "وي", "normalized_value": "WE_TELECOM"})
        elif "ناتجاس" in text:
            entities.append({"entity_type": "biller_name", "raw_text": "ناتجاس", "normalized_value": "NATGAS"})
        elif "جنوب القاهرة" in text:
            entities.append({"entity_type": "biller_name", "raw_text": "جنوب القاهرة", "normalized_value": "SOUTH_CAIRO_ELECTRICITY"})

        if "النت المنزلي" in text or "النت" in text:
            entities.append({"entity_type": "biller_service", "raw_text": "النت المنزلي", "normalized_value": "HOME_INTERNET"})
        elif "الكهرباء" in text or "الكهربا" in text:
            entities.append({"entity_type": "biller_service", "raw_text": "الكهرباء", "normalized_value": "ELECTRICITY"})
        elif "الغاز" in text:
            entities.append({"entity_type": "biller_service", "raw_text": "الغاز", "normalized_value": "GAS"})
        elif "كارت شحن" in text or "شحن كارت" in text:
            entities.append({"entity_type": "biller_service", "raw_text": "كارت شحن", "normalized_value": "MOBILE_RECHARGE"})

        return entities
