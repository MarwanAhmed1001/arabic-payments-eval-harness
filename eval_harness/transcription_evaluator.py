# -*- coding: utf-8 -*-
"""
Transcription Quality & ASR Evaluator for Arabic Payment Voice Instructions.
Computes standard Word Error Rate (WER), Character Error Rate (CER),
Normalized WER (Arabic orthography invariant), and Financial Keyword Error Rate (KER).
"""

from typing import List, Dict, Any, Optional
import Levenshtein
from eval_harness.text_normalizer import normalize_arabic_orthography, normalize_digits

class TranscriptionEvaluator:
    """Evaluates Speech-to-Text (ASR) performance on Arabic and Egyptian dialect instructions."""

    FINANCIAL_KEYWORDS = [
        "جنيه", "جنية", "ج", "ريال", "دولار",
        "حول", "ابعت", "تحويل", "ادفع", "سداد", "اشحن", "استعلام",
        "فودافون", "اورنج", "اتصالات", "وي", "انستاباي", "انستا", "كاش", "محفظة", "ميزة", "اهلي", "مصر",
        "فاتورة", "الكهرباء", "الغاز", "النت", "رصيد"
    ]

    def _compute_wer(self, ref_tokens: List[str], hyp_tokens: List[str]) -> float:
        """Computes Word Error Rate via Levenshtein distance on token lists."""
        if not ref_tokens:
            return 0.0 if not hyp_tokens else 1.0
        dist = Levenshtein.distance(ref_tokens, hyp_tokens)
        return float(dist / len(ref_tokens))

    def evaluate(self, gold_samples: List[Dict[str, Any]], predictions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Evaluates ASR hypotheses against references.
        If predictions is None, uses 'asr_hypothesis' embedded inside the gold sample.
        """
        pred_map = {}
        if predictions:
            pred_map = {p["id"]: p.get("predicted_transcription", "") for p in predictions}

        total_words_raw = 0
        total_word_edits_raw = 0

        total_words_norm = 0
        total_word_edits_norm = 0

        total_chars_raw = 0
        total_char_edits_raw = 0

        total_keywords = 0
        keyword_errors = 0

        sample_details = []

        for sample in gold_samples:
            sample_id = sample["id"]
            ref_raw = sample.get("asr_reference", "")
            if not ref_raw:
                continue

            hyp_raw = pred_map.get(sample_id, sample.get("asr_hypothesis", ""))

            # 1. Raw WER & CER
            ref_words_raw = ref_raw.split()
            hyp_words_raw = hyp_raw.split()

            w_edits_raw = Levenshtein.distance(ref_words_raw, hyp_words_raw)
            c_edits_raw = Levenshtein.distance(ref_raw, hyp_raw)

            total_words_raw += len(ref_words_raw)
            total_word_edits_raw += w_edits_raw
            total_chars_raw += len(ref_raw)
            total_char_edits_raw += c_edits_raw

            sample_wer_raw = w_edits_raw / len(ref_words_raw) if ref_words_raw else 0.0

            # 2. Normalized WER (Arabic orthography & numerals harmonized)
            ref_norm = normalize_arabic_orthography(normalize_digits(ref_raw))
            hyp_norm = normalize_arabic_orthography(normalize_digits(hyp_raw))

            ref_words_norm = ref_norm.split()
            hyp_words_norm = hyp_norm.split()

            w_edits_norm = Levenshtein.distance(ref_words_norm, hyp_words_norm)
            total_words_norm += len(ref_words_norm)
            total_word_edits_norm += w_edits_norm

            sample_wer_norm = w_edits_norm / len(ref_words_norm) if ref_words_norm else 0.0

            # 3. Keyword Error Rate (KER) for critical payment tokens
            ref_keywords = [w for w in ref_words_norm if w in self.FINANCIAL_KEYWORDS or any(char.isdigit() for char in w)]
            hyp_keywords = [w for w in hyp_words_norm if w in self.FINANCIAL_KEYWORDS or any(char.isdigit() for char in w)]

            kw_edits = Levenshtein.distance(ref_keywords, hyp_keywords)
            total_keywords += len(ref_keywords)
            keyword_errors += kw_edits

            sample_details.append({
                "id": sample_id,
                "raw_wer": round(sample_wer_raw, 4),
                "norm_wer": round(sample_wer_norm, 4),
                "ref_raw": ref_raw,
                "hyp_raw": hyp_raw
            })

        overall_raw_wer = total_word_edits_raw / total_words_raw if total_words_raw > 0 else 0.0
        overall_norm_wer = total_word_edits_norm / total_words_norm if total_words_norm > 0 else 0.0
        overall_raw_cer = total_char_edits_raw / total_chars_raw if total_chars_raw > 0 else 0.0
        overall_ker = keyword_errors / total_keywords if total_keywords > 0 else 0.0

        return {
            "total_audio_samples": len(sample_details),
            "raw_wer": round(overall_raw_wer, 4),
            "normalized_wer": round(overall_norm_wer, 4),
            "raw_cer": round(overall_raw_cer, 4),
            "keyword_error_rate_ker": round(overall_ker, 4),
            "orthography_impact_gain": round(overall_raw_wer - overall_norm_wer, 4),
            "sample_evaluations": sample_details[:10]
        }
