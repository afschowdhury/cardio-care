# ❤️ CardioCare AI: Heart Disease Prediction & Clinical Decision Support System

An end-to-end Machine Learning and Explainable AI (XAI) undergraduate thesis project engineered for a **Final Year Project (FYP / Undergraduate Thesis Defense)** in Computer Science, Data Science, Software Engineering, or Biomedical Informatics.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/Package%20Manager-UV-purple.svg)](https://github.com/astral-sh/uv)
[![Framework](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![XAI](https://img.shields.io/badge/Explainability-SHAP-green.svg)](https://github.com/shap/shap)
[![Tests](https://img.shields.io/badge/Unit%20Tests-15%20Passing-brightgreen.svg)](https://docs.pytest.org/)

---

## 🌟 What Was Done Previously vs. What Is New in This Thesis

CardioCare AI directly addresses the five critical flaws found in prior academic ML studies and traditional clinical risk scoring:

1. **Data Hygiene, Target Polarity Canonicalization & Dual-Regime Leakage Audit**:
   - *Prior Work Flaw*: Public Kaggle exports of the 1,025-row UCI Heart Disease CSV (`heart.csv`) contain **inverted target labels** (`0 = Disease`, `1 = Healthy`) and **723 duplicate rows** ($302$ unique patients replicated $\sim 3.4\times$). Uncritical random 80/20 splits leak identical test patients into the training set and invert clinical risk predictions.
   - *What Is New*: Implements automated physiological target polarity verification (`src/data_loader.py`) against exercise ST depression (`oldpeak`) and fluoroscopy vessel stenosis (`ca`), and evaluates **both** the **1,025-record multi-center benchmark** and the **strictly deduplicated 302-patient cohort**, empirically quantifying duplicate-record accuracy inflation ($+3.8\%$ to $+20.0\%$).
2. **Direct Clinical Baseline Comparison (Framingham Risk Score) + Statistical Significance**:
   - *Prior Work Flaw*: Evaluates ML models only against other ML models without testing against clinical hospital calculators.
   - *What Is New*: Implements the standardized adult **Framingham 10-Year Coronary Risk Score** ([`src/framingham.py`](src/framingham.py)), proving that our ensembles significantly outperform Framingham on both the benchmark cohort (**97.07% vs. 61.95% accuracy; 0.9940 vs. 0.6559 ROC-AUC; McNemar $p = 9.44 \times 10^{-15}$**) and the strictly deduplicated cohort (**83.61% vs. 60.66% accuracy; 0.8853 vs. 0.6402 ROC-AUC; McNemar $p = 0.0080$**).
3. **Leakage-Free Out-of-Fold Cost-Sensitive Threshold Optimization**:
   - *Prior Work Flaw*: Uses an arbitrary symmetric $\tau = 0.50$ threshold (or tunes thresholds on the test set).
   - *What Is New*: Formulates an asymmetric clinical cost matrix ($C_{FN} = 5.0 \cdot C_{FP}$) optimized via 5-fold **Out-of-Fold (OOF)** cross-validation on the training partition ([`src/threshold_optimizer.py`](src/threshold_optimizer.py)), identifying $\tau^* = 0.20$ and reducing clinical penalty cost by **38.9% for GBM** and **53.7% for Random Forest** while achieving **100.0% test sensitivity**.
4. **Schema-Aligned Cooperative Game-Theoretic Explainability (SHAP)**:
   - *Prior Work Flaw*: Operates as a black box or scrambles feature attributions when `ColumnTransformer` reorders numeric and categorical columns.
   - *What Is New*: Aligns **TreeSHAP** (`src/explain.py`) strictly with `ColumnTransformer.get_feature_names_out()`, generating patient-specific waterfall charts and global rankings that match cardiovascular pathophysiology.
5. **Monotonic Multi-Tier "What-If" Counterfactual Intervention Simulator**:
   - *Prior Work Flaw*: Stops at passive diagnosis (*"High Risk"*) without actionable treatment targets.
   - *What Is New*: Separates immutable traits (`age`, `sex`) from modifiable hemodynamic, lipid, and ischemic markers ([`src/counterfactual.py`](src/counterfactual.py)), computing trajectory-optimized 3-tier therapeutic plans that reduce a high-risk patient's predicted probability from **99.3% down to 13.7% (Low Risk < 30%)**.

---

## 📊 Experimental Benchmark Results

### Regime 1: Multi-Center Benchmark Cohort ($N = 1,025$ Total; $N_{\text{test}} = 205$)

| Model / Algorithm | Test Accuracy | Recall (Sensitivity) | Precision | Specificity | F1-Score | ROC-AUC | Brier Score | 5-Fold CV Accuracy | McNemar $p$ (vs. FRS) |
|---|---|---|---|---|---|---|---|---|---|
| **Framingham Risk Score (Clinical Baseline)** | **61.95%** | **54.00%** | **61.36%** | **69.52%** | **57.45%** | **0.6559** | **0.3510** | **N/A (Heuristic)** | **Ref (1.00)** |
| **Logistic Regression** | 82.44% | 75.00% | 87.21% | 89.52% | 80.65% | 0.9134 | 0.1174 | 83.78% ± 1.62% | $9.82 \times 10^{-6}$ |
| **Decision Tree** | 88.29% | 83.00% | 92.22% | 93.33% | 87.37% | 0.9444 | 0.0861 | 87.93% ± 1.90% | $4.83 \times 10^{-9}$ |
| **Support Vector Machine (SVM)** | 88.29% | 84.00% | 91.30% | 92.38% | 87.50% | 0.9560 | 0.0816 | 87.68% ± 2.48% | $1.61 \times 10^{-8}$ |
| **Random Forest** | 95.61% | **100.00%** | 91.74% | 91.43% | 95.69% | 0.9920 | 0.0353 | 98.17% ± 1.22% | $4.17 \times 10^{-14}$ |
| **Gradient Boosting (GBM)** | **97.07%** | **100.00%** | **94.34%** | **94.29%** | **97.09%** | **0.9940** | **0.0280** | **98.54% ± 0.99%** | **$9.44 \times 10^{-15}$** |

### Regime 2: Strict Deduplicated Patient-Disjoint Cohort ($N = 302$ Unique Patients; $N_{\text{test}} = 61$)

| Model / Algorithm | Dedup Accuracy | Dedup Recall | Dedup Precision | Dedup F1 | Dedup ROC-AUC | Brier Score | McNemar $p$ (vs. FRS) |
|---|---|---|---|---|---|---|---|
| **Framingham Risk Score (Clinical Baseline)** | **60.66%** | **53.57%** | **57.69%** | **55.56%** | **0.6402** | **0.3519** | **Ref (1.00)** |
| **Logistic Regression** | 78.69% | 75.00% | 77.78% | 76.36% | **0.8853** | 0.1369 | 0.0455 |
| **Decision Tree** | 77.05% | 75.00% | 75.00% | 75.00% | 0.8366 | 0.1808 | 0.0662 |
| **Gradient Boosting (GBM)** | 77.05% | 78.57% | 73.33% | 75.86% | 0.8647 | 0.1717 | 0.0662 |
| **Random Forest** | 81.97% | **82.14%** | 79.31% | 80.70% | **0.8853** | 0.1388 | 0.0123 |
| **Support Vector Machine (SVM)** | **83.61%** | 75.00% | **87.50%** | **80.77%** | 0.8755 | **0.1316** | **0.0080** |

---

## 🎓 Academic Thesis Documents (LaTeX, PDF & Markdown)

- **LaTeX Thesis Report Source**: [`report/thesis_report.tex`](report/thesis_report.tex)
- **Compiled PDF Thesis Report**: [`report/thesis_report.pdf`](report/thesis_report.pdf)
- **Publication Figures (Vector PDF & PNG)**: [`report/figures/`](report/figures/)
- **Markdown Capstone Report**: [`report/FYP_Final_Report.md`](report/FYP_Final_Report.md)
- **Viva Voce Defense & Slide Guide**: [`report/viva_presentation_guide.md`](report/viva_presentation_guide.md)

---

## 🚀 Getting Started

Run every command from the **repository root**. The `src/` package is imported via `sys.path` relative to that directory.

### Prerequisites

- **Python 3.9 or newer** (developed with CPython 3.9.6)
- **Git**
- Optional: [uv](https://github.com/astral-sh/uv) (recommended; this project’s local environment was created with uv)
- Optional: TeX Live / MacTeX if you want to recompile [`report/thesis_report.tex`](report/thesis_report.tex)
- **macOS + XGBoost**: install OpenMP first, or training will fall back to scikit-learn’s `GradientBoostingClassifier`:

```bash
brew install libomp
```

### 1. Clone the repository

```bash
git clone <your-repo-url> cardiocare-ai
cd cardiocare-ai
```

### 2. Create a virtual environment and install dependencies

**Recommended (uv)**

```bash
uv venv --python 3.9
source .venv/bin/activate
uv pip install -r requirements.txt
```

**Fallback (stdlib venv + pip)**

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

After activation, `python`, `pytest`, and `streamlit` all refer to the environment. You can also invoke them as `./.venv/bin/python`, `./.venv/bin/pytest`, and `./.venv/bin/streamlit` without activating.

### 3. Train models (optional)

Trained artifacts already live in [`models/`](models/) (`best_model.joblib`, per-algorithm `.joblib` files, `preprocessor.joblib`, and `model_metrics.json`). Skip this step unless you want to regenerate them.

```bash
python src/train.py
```

This runs dual-regime training, 5-fold cross-validation, Framingham comparison, and out-of-fold threshold optimization, then overwrites files in `models/`.

### 4. Run the unit tests

```bash
pytest tests/ -v
```

Some tests load the saved models. If those artifacts are missing, run `python src/train.py` first.

### 5. Launch the Streamlit clinician dashboard

```bash
streamlit run app/app.py
```

Open the URL Streamlit prints (typically [http://localhost:8501](http://localhost:8501)).

### 6. Recompile the LaTeX thesis report (optional)

Requires TeX Live or MacTeX. Run `pdflatex` twice so the table of contents and cross-references resolve:

```bash
cd report
pdflatex thesis_report.tex
pdflatex thesis_report.tex
```

A compiled PDF is already committed at [`report/thesis_report.pdf`](report/thesis_report.pdf).

---

## 📁 Project Structure

```
cardiocare-ai/
├── app/
│   ├── app.py              Streamlit clinician dashboard
│   └── style.css
├── data/raw/
│   ├── heart.csv           1,025-row UCI / Kaggle export
│   └── heart_cleveland_303.csv
├── models/                 Trained .joblib artifacts + model_metrics.json
├── src/
│   ├── data_loader.py      Load, polarity check, deduplicate
│   ├── preprocess.py       ColumnTransformer + train/test split
│   ├── train.py            Dual-regime training and evaluation
│   ├── framingham.py       Clinical baseline
│   ├── threshold_optimizer.py
│   ├── explain.py          TreeSHAP
│   └── counterfactual.py   What-if intervention simulator
├── tests/                  pytest suite (15 tests)
├── report/                 LaTeX, PDF, figures, and markdown write-ups
└── requirements.txt
```

---

## 🛠️ Troubleshooting

| Problem | What to do |
|---|---|
| `FileNotFoundError: Best model not found ... Please run train.py first.` | The dashboard and some tests need [`models/best_model.joblib`](models/best_model.joblib). Run `python src/train.py` from the repo root. |
| `ModuleNotFoundError: No module named 'src'` | You are not in the repository root. `cd` to the folder that contains `src/` and `app/`, then rerun the command. |
| XGBoost fails on macOS (`libomp` / OpenMP) | `brew install libomp`, or ignore it: `src/train.py` catches the failure and trains scikit-learn Gradient Boosting instead. |
| Streamlit cannot start | Confirm the venv is active (`which streamlit` should point at `.venv`) and that you installed `requirements.txt`. |
| Tests fail before training | `pytest tests/test_data.py` can run without models; `test_prediction.py` and `test_improvements.py` need the committed (or freshly trained) artifacts in `models/`. |
