# -*- coding: utf-8 -*-
"""
Machine Learning Intent Classifier for Egyptian Dialect Payments.
Trains a statistical NLP classifier (TF-IDF N-grams + Linear SVM / Logistic Regression)
on payment instructions, demonstrating end-to-end ML training, evaluation, and inference.
"""

from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from models.base_runner import BasePaymentModelRunner
from eval_harness.text_normalizer import normalize_arabic_orthography

# Training corpus for Egyptian dialect payment intents
TRAIN_PAYMENT_CORPUS = [
    # P2P_TRANSFER
    ("حول 500 جنيه لمحمد على فودافون كاش", "P2P_TRANSFER"),
    ("ابعت الف جنيه لاحمد على انستاباي", "P2P_TRANSFER"),
    ("عايز احول فلوس لمحفظة اتصالات كاش", "P2P_TRANSFER"),
    ("ممكن تحويل 300 جنيه لحساب البنك الاهلي", "P2P_TRANSFER"),
    ("حول لمصطفى تمن الحاجة ميتين جنيه", "P2P_TRANSFER"),
    ("ابعت باكو لاخويا على اورنج كاش", "P2P_TRANSFER"),
    ("حول 1500 ج لمحفظة فودافون كاش", "P2P_TRANSFER"),
    ("ابعت فلوس على انستا باي للحساب ده", "P2P_TRANSFER"),
    ("عايز ابعت 200 جنيه لصاحبي", "P2P_TRANSFER"),

    # BILL_PAYMENT
    ("ادفع فاتورة النت المنزلي لشركة وي", "BILL_PAYMENT"),
    ("سداد فاتورة الكهرباء جنوب القاهرة 450 جنيه", "BILL_PAYMENT"),
    ("اشحن كارت رصيد فودافون بمية جنيه", "BILL_PAYMENT"),
    ("عايز اسدد فاتورة الغاز الطبيعي ناتجاس", "BILL_PAYMENT"),
    ("ادفع فاتورة التليفون الارضي وي", "BILL_PAYMENT"),
    ("شحن رصيد اورنج بمبلغ خمسين جنيه", "BILL_PAYMENT"),
    ("فكلي كارت شحن بعشرينيه لرقمي", "BILL_PAYMENT"),
    ("سدد الفاتورة بتاعة الكهربا", "BILL_PAYMENT"),

    # BALANCE_INQUIRY
    ("هو رصيدي الحالي في المحفظة كام يا فندم", "BALANCE_INQUIRY"),
    ("عايز استعلم عن متبقي حساب البنك والفلوس", "BALANCE_INQUIRY"),
    ("رصيد كارت ميزة بتاعي كام دلوقتي", "BALANCE_INQUIRY"),
    ("فاضل معايا كام في الحساب", "BALANCE_INQUIRY"),
    ("استعلام عن الرصيد المتبقي في فودافون كاش", "BALANCE_INQUIRY"),
    ("معايا كام فلوس في المحفظة", "BALANCE_INQUIRY"),

    # OUT_OF_SCOPE
    ("الفيزا بتاعتي اتسحبت في مكنة الـ ATM في المعادي", "OUT_OF_SCOPE"),
    ("مساء الخير اخبارك ايه وايه عروض باقات وي", "OUT_OF_SCOPE"),
    ("عايز اغير كلمة السر بتاعة التطبيق عشان نسيتها", "OUT_OF_SCOPE"),
    ("أنا اشتريت غسالة وعايز اعرف مواعيد الشحن عندكم", "OUT_OF_SCOPE"),
    ("الراوتر فاصل بقاله يومين والنت مش شغال", "OUT_OF_SCOPE"),
    ("عايز اقدم شكوى في خدمة العملاء وموظف الفرع", "OUT_OF_SCOPE"),
    ("هو الفرع بيفتح الساعة كام يوم الجمعة", "OUT_OF_SCOPE"),
    ("لا خلاص الغي التحويل اوعى تبعت حاجة", "OUT_OF_SCOPE")
]

class MLIntentRunner(BasePaymentModelRunner):
    """Machine Learning model trained on Egyptian dialect payment corpus."""

    def __init__(self):
        super().__init__(model_name="TFIDF-LinearClassifier-v1.0")
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                preprocessor=normalize_arabic_orthography,
                ngram_range=(1, 3),
                sublinear_tf=True
            )),
            ("clf", LogisticRegression(C=5.0, max_iter=200, class_weight="balanced"))
        ])
        self._train()

    def _train(self):
        texts = [x[0] for x in TRAIN_PAYMENT_CORPUS]
        labels = [x[1] for x in TRAIN_PAYMENT_CORPUS]
        self.pipeline.fit(texts, labels)

    def predict_intent(self, text: str) -> str:
        return self.pipeline.predict([text])[0]

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        # Uses entity resolution on top of ML predictions
        from models.semantic_hybrid_runner import SemanticHybridRunner
        fallback_extractor = SemanticHybridRunner()
        return fallback_extractor.extract_entities(text)
