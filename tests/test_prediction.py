"""
Unit Tests for Model Inference, Explainability, Dual-Regime Metrics, and Artifact Integrity
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.explain import CardioCareExplainer
from src.preprocess import load_preprocessor, transform_single_patient
from src.train import load_best_model


@pytest.fixture
def sample_patient():
    return {
        "age": 67,
        "sex": 1,
        "cp": 0,
        "trestbps": 168,
        "chol": 298,
        "fbs": 1,
        "restecg": 2,
        "thalach": 112,
        "exang": 1,
        "oldpeak": 3.4,
        "slope": 1,
        "ca": 2,
        "thal": 3
    }


def test_saved_artifacts_exist():
    models_dir = os.path.join(PROJECT_ROOT, "models")
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    prep_path = os.path.join(models_dir, "preprocessor.joblib")
    metrics_path = os.path.join(models_dir, "model_metrics.json")

    assert os.path.exists(best_model_path), "best_model.joblib does not exist"
    assert os.path.exists(prep_path), "preprocessor.joblib does not exist"
    assert os.path.exists(metrics_path), "model_metrics.json does not exist"


def test_model_inference_probability_and_clinical_polarity(sample_patient):
    best_model, _ = load_best_model()
    preprocessor = load_preprocessor()

    healthy_patient = {
        "age": 32, "sex": 0, "cp": 1, "trestbps": 115, "chol": 175,
        "fbs": 0, "restecg": 1, "thalach": 178, "exang": 0, "oldpeak": 0.2,
        "slope": 2, "ca": 0, "thal": 2
    }

    X_senior = transform_single_patient(sample_patient, preprocessor)
    X_healthy = transform_single_patient(healthy_patient, preprocessor)

    prob_senior = float(best_model.predict_proba(X_senior)[0, 1])
    prob_healthy = float(best_model.predict_proba(X_healthy)[0, 1])

    assert 0.0 <= prob_senior <= 1.0
    assert 0.0 <= prob_healthy <= 1.0
    assert prob_senior > 0.80, "High-risk hypertensive senior should have >80% predicted risk"
    assert prob_healthy < 0.15, "Healthy young adult should have <15% predicted risk"


def test_model_metrics_json_integrity():
    metrics_path = os.path.join(PROJECT_ROOT, "models", "model_metrics.json")
    with open(metrics_path, "r") as f:
        data = json.load(f)

    assert "best_model_name" in data
    assert "models" in data
    assert "deduplicated_models" in data
    assert "threshold_optimization" in data
    assert data["threshold_optimization"]["cost_reduction_pct"] > 0.0

    for mname, mval in data["models"].items():
        assert "test_accuracy" in mval
        assert "test_recall" in mval
        assert "test_f1" in mval
        assert "test_brier_score" in mval
        assert "confusion_matrix" in mval
        assert "roc_curve" in mval
        assert 0.0 <= mval["test_accuracy"] <= 1.0


def test_shap_explainer_feature_alignment(sample_patient):
    explainer = CardioCareExplainer()
    explanation = explainer.explain_patient(sample_patient)

    assert "shap_values" in explanation
    assert len(explanation["shap_values"]) == 13
    assert "summary_bullets" in explanation
    assert len(explanation["summary_bullets"]) > 0

    # Verify that ST depression (oldpeak=3.4) and fluoroscopy vessels (ca=2) have positive risk attribution
    impact_map = {item["feature"]: item["impact"] for item in explanation["impacts_sorted"]}
    assert impact_map["oldpeak"] > 0, "Severe ST depression (3.4 mm) must have positive SHAP risk impact"
    assert impact_map["ca"] > 0, "Stenotic fluoroscopy vessels (ca=2) must have positive SHAP risk impact"

    fig = explainer.generate_waterfall_plot(sample_patient)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
