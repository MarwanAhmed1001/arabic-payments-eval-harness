# -*- coding: utf-8 -*-
"""
Generate 1-Page ATS-Compliant Resume in Markdown and DOCX format.
"""

from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

RESUME_MD = """# MARWAN AHMED
**AI & Machine Learning Engineer | Conversational AI & NLP Evaluations**  
Cairo, Egypt | (+20) 12 12451743 | marwanshehata328@gmail.com  
[GitHub: MarwanAhmed1001](https://github.com/MarwanAhmed1001) | [LinkedIn](https://linkedin.com/in/marwan-ahmed-173549374) | [Portfolio: personal-one.netlify.app](https://personal-one.netlify.app)

---

### PROFESSIONAL SUMMARY
AI & Machine Learning Engineer with strong foundations in Natural Language Processing, model evaluation harness design, and predictive modeling. Experienced in building reproducible Python evaluation suites for Arabic and Egyptian dialect conversational systems, mitigating data leakage, and benchmarking classification and deep learning models. Committed to rigorous measurement, honest empirical reporting, and production-ready code.

---

### TECHNICAL SKILLS
* **Languages & Core:** Python (Modular Scripts, CLI, PEP8), SQL, C++, Bash/Linux
* **ML & Deep Learning:** PyTorch, Scikit-Learn, TensorFlow, Transformers, Pandas, NumPy, Scipy
* **NLP & Evaluation:** Evaluation Harness Design, Intent Classification, Named Entity Recognition (NER), Slot Error Rate (SER), Word Error Rate (WER/CER), Exact Match (EM), Text Normalization, TF-IDF
* **Methodologies & Hygiene:** Train/Test Splitting (`GroupKFold`), Data Leakage Prevention, Cross-Validation, Class Imbalance Handling, Regression & Error Analysis, Git (Branches, Diffs, PRs)
* **Deployment & Tools:** Streamlit, REST APIs, Git/GitHub, Pydantic, Virtualenv

---

### CORE MACHINE LEARNING & EVALUATION PROJECTS

#### 1. Arabic & Egyptian Dialect Payment Evaluation Harness
*Technologies: Python, Scikit-Learn, RapidFuzz, Levenshtein, CLI, Pydantic* | [GitHub Repository](https://github.com/MarwanAhmed1001/arabic-payments-eval-harness)
* Engineered an automated evaluation harness and versioned benchmark suite measuring intent classification, financial slot extraction, and dialect ASR transcription across Egyptian payment instructions (InstaPay, Vodafone Cash, utility bills).
* Formulated domain-specific metrics including **Out-of-Scope (OOS) False Acceptance Rate** to prevent catastrophic financial execution on complaints, and **Value-Normalized Matching** for colloquial numbers (*"مية وخمسين"*, *"باكو"*).
* Built an automated regression tracker comparing candidate models against baselines, flagging regressions and achieving **+55.5% accuracy gain** and **0% false acceptance** on adversarial stress tests.
* Packaged as cloneable, modular Python CLI scripts (`evaluate.py`) with zero-dependency unit tests and automated executive findings documentation (`FINDINGS.md`).

#### 2. Telecom Customer Churn Prediction & Retention Pipeline
*Technologies: Python, Scikit-Learn, Pandas, Feature Engineering, Imbalanced-Learn*
* Built an end-to-end supervised classification pipeline forecasting customer attrition to protect recurring revenue for a telecom provider.
* Resolved severe class imbalance using SMOTE and cost-sensitive class weights, tuning decision thresholds to minimize costly False Negatives.
* Evaluated classifiers across ROC-AUC (0.86), Precision-Recall curves, and business ROI matrices rather than raw accuracy.

#### 3. Vision Transformer (ViT) Arabic Sign Language Recognition
*Technologies: Python, PyTorch, Vision Transformers, Computer Vision, OpenCV*
* Developed a deep learning vision system classifying Arabic Sign Language gestures across multi-class image datasets.
* Implemented transfer learning with a Vision Transformer (ViT) architecture, applying regularization and data augmentation (rotation, scaling, noise injection) to prevent overfitting.
* Monitored training dynamics (loss curves, learning rate scheduling, validation convergence) to ensure generalization on unseen test splits.

#### 4. Content-Based Information Retrieval & Recommender Engine
*Technologies: Python, NLP, Scikit-Learn, TF-IDF, Cosine Similarity, REST APIs, Streamlit*
* Built an NLP retrieval and recommendation engine matching textual user preferences against high-dimensional item synopses.
* Implemented TF-IDF vectorization and Cosine Similarity, mitigating the cold-start problem via hybrid popularity weighting.
* Evaluated recommendation quality via Precision@K, Recall@K, and MAP@K; deployed an interactive Streamlit application integrated with the Jikan REST API.

---

### EDUCATION
**Bachelor of Medical Informatics & Artificial Intelligence** | Alreyada University for Science and Technology  
*Expected Graduation: 2027 | Cairo, Egypt*  
*Relevant Coursework:* Natural Language Processing, Machine Learning, Deep Learning, Probability & Statistics, Data Structures, Algorithms.

---

### PROFESSIONAL TRAINING & CERTIFICATIONS
* **Data Science & ML Specialist Trainee** – Digital Egypt Pioneers Initiative (DEPI) *(2024 – 2025)*  
  Completed hands-on training in statistical modeling, machine learning workflows, and data pipelines.
* **Certified Data Scientist & Machine Learning Specialist** – DEPI & Official Partners

---

### LANGUAGES
* **Arabic:** Native (Egyptian Dialect & Modern Standard Arabic) | **English:** Professional Working Proficiency
"""

def generate_docx(output_path: Path):
    doc = docx.Document()

    # Set 0.5 inch margins for 1-page fit
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.4)
        section.bottom_margin = Inches(0.4)
        section.left_margin = Inches(0.5)
        section.right_margin = Inches(0.5)

    # Base style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(9.5)
    normal_style.font.color.rgb = RGBColor(30, 30, 30)

    # Header
    p_name = doc.add_paragraph()
    p_name.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_name.paragraph_format.space_before = Pt(0)
    p_name.paragraph_format.space_after = Pt(2)
    run_name = p_name.add_run("MARWAN AHMED")
    run_name.bold = True
    run_name.font.size = Pt(16)
    run_name.font.color.rgb = RGBColor(10, 45, 90)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(2)
    run_title = p_title.add_run("AI & Machine Learning Engineer | Conversational AI & NLP Evaluations")
    run_title.bold = True
    run_title.font.size = Pt(10)

    p_contact = doc.add_paragraph()
    p_contact.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_contact.paragraph_format.space_after = Pt(6)
    p_contact.add_run("Cairo, Egypt | (+20) 12 12451743 | marwanshehata328@gmail.com\n")
    p_contact.add_run("GitHub: github.com/MarwanAhmed1001 | LinkedIn: linkedin.com/in/marwan-ahmed-173549374 | Portfolio: personal-one.netlify.app")

    def add_section_heading(title: str):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(title.upper())
        run.bold = True
        run.font.size = Pt(10.5)
        run.font.color.rgb = RGBColor(10, 45, 90)
        # Add bottom border via xml
        pBdr = parse_xml(r'<w:pBdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                         r'<w:bottom w:val="single" w:sz="6" w:space="1" w:color="0A2D5A"/>'
                         r'</w:pBdr>')
        p._p.get_or_add_pPr().append(pBdr)

    # 1. Professional Summary
    add_section_heading("Professional Summary")
    p_sum = doc.add_paragraph()
    p_sum.paragraph_format.space_after = Pt(4)
    p_sum.add_run(
        "AI & Machine Learning Engineer with strong foundations in Natural Language Processing, model evaluation harness design, "
        "and predictive modeling. Experienced in building reproducible Python evaluation suites for Arabic and Egyptian dialect conversational systems, "
        "mitigating data leakage, and benchmarking classification and deep learning models. Committed to rigorous measurement, honest empirical reporting, and production-ready code."
    )

    # 2. Technical Skills
    add_section_heading("Technical Skills")
    skills = [
        ("Languages & Core: ", "Python (Modular Scripts, CLI, PEP8), SQL, C++, Bash/Linux"),
        ("ML & Deep Learning: ", "PyTorch, Scikit-Learn, TensorFlow, Transformers, Pandas, NumPy, Scipy"),
        ("NLP & Evaluation: ", "Evaluation Harness Design, Intent Classification, NER, Slot Error Rate (SER), Word Error Rate (WER/CER), Exact Match (EM), Text Normalization, TF-IDF"),
        ("Methodologies & Hygiene: ", "Train/Test Splitting (GroupKFold), Data Leakage Prevention, Cross-Validation, Class Imbalance Handling, Regression & Error Analysis, Git"),
        ("Deployment & Tools: ", "Streamlit, REST APIs, Git/GitHub, Pydantic, Virtualenv")
    ]
    for cat, items in skills:
        p_sk = doc.add_paragraph()
        p_sk.paragraph_format.space_before = Pt(0)
        p_sk.paragraph_format.space_after = Pt(1)
        r_cat = p_sk.add_run("• " + cat)
        r_cat.bold = True
        p_sk.add_run(items)

    # 3. Core Projects
    add_section_heading("Core Machine Learning & Evaluation Projects")

    # Project 1
    p_pr1 = doc.add_paragraph()
    p_pr1.paragraph_format.space_before = Pt(3)
    p_pr1.paragraph_format.space_after = Pt(1)
    r_pr1 = p_pr1.add_run("Arabic & Egyptian Dialect Payment Evaluation Harness")
    r_pr1.bold = True
    p_pr1.add_run(" | Python, Scikit-Learn, RapidFuzz, Levenshtein, CLI | github.com/MarwanAhmed1001/arabic-payments-eval-harness")

    bullets_pr1 = [
        "Engineered an automated evaluation harness and versioned benchmark suite measuring intent classification, financial slot extraction, and dialect ASR transcription across Egyptian payment instructions (InstaPay, Vodafone Cash, utility bills).",
        "Formulated domain-specific metrics including Out-of-Scope (OOS) False Acceptance Rate to prevent catastrophic financial execution on complaints, and Value-Normalized Matching for colloquial numbers (\"مية وخمسين\", \"باكو\").",
        "Built an automated regression tracker comparing candidate models against baselines, flagging regressions and achieving +55.5% accuracy gain and 0% false acceptance on adversarial stress tests.",
        "Packaged as cloneable, modular Python CLI scripts (evaluate.py) with zero-dependency unit tests and automated executive findings documentation (FINDINGS.md)."
    ]
    for b in bullets_pr1:
        bp = doc.add_paragraph()
        bp.paragraph_format.space_before = Pt(0)
        bp.paragraph_format.space_after = Pt(1)
        bp.paragraph_format.left_indent = Inches(0.2)
        bp.add_run("• " + b)

    # Project 2
    p_pr2 = doc.add_paragraph()
    p_pr2.paragraph_format.space_before = Pt(3)
    p_pr2.paragraph_format.space_after = Pt(1)
    r_pr2 = p_pr2.add_run("Telecom Customer Churn Prediction & Retention Pipeline")
    r_pr2.bold = True
    p_pr2.add_run(" | Python, Scikit-Learn, Pandas, Feature Engineering, Imbalanced-Learn")

    bullets_pr2 = [
        "Built an end-to-end supervised classification pipeline forecasting customer attrition to protect recurring revenue for a telecom provider.",
        "Resolved severe class imbalance using SMOTE and cost-sensitive class weights, tuning decision thresholds to minimize costly False Negatives.",
        "Evaluated classifiers across ROC-AUC (0.86), Precision-Recall curves, and business ROI matrices rather than raw accuracy."
    ]
    for b in bullets_pr2:
        bp = doc.add_paragraph()
        bp.paragraph_format.space_before = Pt(0)
        bp.paragraph_format.space_after = Pt(1)
        bp.paragraph_format.left_indent = Inches(0.2)
        bp.add_run("• " + b)

    # Project 3
    p_pr3 = doc.add_paragraph()
    p_pr3.paragraph_format.space_before = Pt(3)
    p_pr3.paragraph_format.space_after = Pt(1)
    r_pr3 = p_pr3.add_run("Vision Transformer (ViT) Arabic Sign Language Recognition")
    r_pr3.bold = True
    p_pr3.add_run(" | Python, PyTorch, Vision Transformers, Computer Vision, OpenCV")

    bullets_pr3 = [
        "Developed a deep learning vision system classifying Arabic Sign Language gestures across multi-class image datasets.",
        "Implemented transfer learning with a Vision Transformer (ViT) architecture, applying regularization and data augmentation (rotation, scaling, noise injection) to prevent overfitting.",
        "Monitored training dynamics (loss curves, learning rate scheduling, validation convergence) to ensure generalization on unseen test splits."
    ]
    for b in bullets_pr3:
        bp = doc.add_paragraph()
        bp.paragraph_format.space_before = Pt(0)
        bp.paragraph_format.space_after = Pt(1)
        bp.paragraph_format.left_indent = Inches(0.2)
        bp.add_run("• " + b)

    # Project 4
    p_pr4 = doc.add_paragraph()
    p_pr4.paragraph_format.space_before = Pt(3)
    p_pr4.paragraph_format.space_after = Pt(1)
    r_pr4 = p_pr4.add_run("Content-Based Information Retrieval & Recommender Engine")
    r_pr4.bold = True
    p_pr4.add_run(" | Python, NLP, Scikit-Learn, TF-IDF, Cosine Similarity, REST APIs, Streamlit")

    bullets_pr4 = [
        "Built an NLP retrieval engine matching user preferences against high-dimensional item synopses using TF-IDF and Cosine Similarity.",
        "Mitigated the cold-start problem via hybrid popularity weighting and evaluated ranking quality via Precision@K, Recall@K, and MAP@K; deployed an interactive Streamlit application."
    ]
    for b in bullets_pr4:
        bp = doc.add_paragraph()
        bp.paragraph_format.space_before = Pt(0)
        bp.paragraph_format.space_after = Pt(1)
        bp.paragraph_format.left_indent = Inches(0.2)
        bp.add_run("• " + b)

    # 4. Education
    add_section_heading("Education")
    p_ed = doc.add_paragraph()
    p_ed.paragraph_format.space_before = Pt(2)
    p_ed.paragraph_format.space_after = Pt(1)
    r_ed = p_ed.add_run("Bachelor of Medical Informatics & Artificial Intelligence")
    r_ed.bold = True
    p_ed.add_run(" | Alreyada University for Science and Technology (Expected: 2027)")
    p_ed2 = doc.add_paragraph()
    p_ed2.paragraph_format.space_before = Pt(0)
    p_ed2.paragraph_format.space_after = Pt(2)
    p_ed2.paragraph_format.left_indent = Inches(0.2)
    p_ed2.add_run("• Relevant Coursework: Natural Language Processing, Machine Learning, Deep Learning, Probability & Statistics, Data Structures, Algorithms.")

    # 5. Training & Certifications
    add_section_heading("Professional Training & Certifications")
    p_tr = doc.add_paragraph()
    p_tr.paragraph_format.space_before = Pt(2)
    p_tr.paragraph_format.space_after = Pt(1)
    r_tr = p_tr.add_run("Data Science & ML Specialist Trainee")
    r_tr.bold = True
    p_tr.add_run(" – Digital Egypt Pioneers Initiative (DEPI) (2024 – 2025)")
    p_tr2 = doc.add_paragraph()
    p_tr2.paragraph_format.space_before = Pt(0)
    p_tr2.paragraph_format.space_after = Pt(2)
    p_tr2.paragraph_format.left_indent = Inches(0.2)
    p_tr2.add_run("• Completed intensive training in statistical modeling, feature engineering pipelines, and strict train/test split hygiene.")

    # Languages
    add_section_heading("Languages")
    p_lang = doc.add_paragraph()
    p_lang.paragraph_format.space_before = Pt(2)
    p_lang.paragraph_format.space_after = Pt(2)
    p_lang.add_run("Arabic: Native (Egyptian Dialect & Modern Standard Arabic) | English: Professional Working Proficiency")

    doc.save(output_path)
    print(f"Generated DOCX Resume -> {output_path}")

def main():
    resume_dir = Path("d:/intern/resume")
    resume_dir.mkdir(parents=True, exist_ok=True)

    md_file = resume_dir / "ATS_Resume_Marwan_Ahmed.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(RESUME_MD.strip())
    print(f"Generated Markdown Resume -> {md_file}")

    docx_file = resume_dir / "ATS_Resume_Marwan_Ahmed.docx"
    generate_docx(docx_file)

if __name__ == "__main__":
    main()
