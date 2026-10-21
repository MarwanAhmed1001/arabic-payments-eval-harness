# -*- coding: utf-8 -*-
"""
Abstract Base Runner for Arabic Payment Conversational Models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BasePaymentModelRunner(ABC):
    """Abstract interface for payment model inference."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def predict_intent(self, text: str) -> str:
        """Predicts high-level intent from Arabic text."""
        pass

    @abstractmethod
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extracts financial entities and normalized values."""
        pass

    def run_inference(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Runs full turn prediction for a single test sample."""
        text = sample["text"]
        pred_intent = self.predict_intent(text)
        pred_entities = self.extract_entities(text)

        return {
            "id": sample["id"],
            "text": text,
            "predicted_intent": pred_intent,
            "predicted_entities": pred_entities
        }

    def run_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes a batch of evaluation samples."""
        return [self.run_inference(s) for s in samples]
