# -*- coding: utf-8 -*-
"""
Arabic & Egyptian Dialect Text Normalization Utilities.
Ensures uniform text comparison across Arabic script variants and numerals.
"""

import re
import unicodedata

# Mapping for Arabic-Indic numerals to standard ASCII digits
ARABIC_INDIC_DIGITS = {
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
}

# Arabic diacritics regex
TASHKEEL_REGEX = re.compile(r'[\u064B-\u0652\u0653-\u0655\u0670]')
TATWEEL_REGEX = re.compile(r'\u0640')

def normalize_digits(text: str) -> str:
    """Converts Eastern Arabic numerals (٠-٩) to Western Arabic digits (0-9)."""
    if not text:
        return ""
    for ar_digit, ascii_digit in ARABIC_INDIC_DIGITS.items():
        text = text.replace(ar_digit, ascii_digit)
    return text

def strip_tashkeel(text: str) -> str:
    """Strips Arabic diacritics (harakat / tashkeel)."""
    return TASHKEEL_REGEX.sub('', text)

def strip_tatweel(text: str) -> str:
    """Strips Arabic elongation character (kashida)."""
    return TATWEEL_REGEX.sub('', text)

def normalize_arabic_orthography(text: str, harmonize_taa_marbouta: bool = True) -> str:
    """
    Standardizes Arabic orthography:
    - Normalizes Alef forms (أ, إ, آ, ٱ -> ا)
    - Normalizes Alef Maksoura (ى -> ي)
    - Optionally normalizes Taa Marbouta (ة -> ه) for dialect phonetic matching
    """
    if not text:
        return ""
    
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    text = strip_tashkeel(text)
    text = strip_tatweel(text)
    text = normalize_digits(text)

    # Alef variants
    text = re.sub(r'[أإآٱ]', 'ا', text)
    # Yaa / Alef Maksoura
    text = re.sub(r'[ى]', 'ي', text)

    if harmonize_taa_marbouta:
        text = re.sub(r'[ة]', 'ه', text)

    # Clean redundant punctuation and extra spaces
    text = re.sub(r'[،؛؟\?!\.,;:\-_/\\]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_egyptian_phone(phone_str: str) -> str:
    """Normalizes Egyptian mobile and landline numbers (removes +2, 002, dashes, spaces)."""
    digits = re.sub(r'\D', '', normalize_digits(phone_str))
    if digits.startswith('20'):
        digits = digits[2:]
    elif digits.startswith('0020'):
        digits = digits[4:]
    if not digits.startswith('0') and len(digits) in (10, 9):
        digits = '0' + digits
    return digits
