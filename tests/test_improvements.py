"""
Unit Tests for Project Improvements:
Framingham Baseline, Cost-Sensitive Threshold Optimization, and What-If Counterfactuals
"""

import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.counterfactual import (
    generate_actionable_plans,
    predict_patient_risk,
    simulate_intervention,
)
from src.framingham import (
    calculate_framingham_risk,
    evaluate_framingham_on_dataframe,
)
from src.preprocess import load_preprocessor
from src.threshold_optimizer import (
    compute_clinical_cost,
    find_optimal_threshold,
)
from src.train import load_best_model


def test_framingham_risk_validity():
    male_patient = {"age": 60, "sex": 1, "trestbps": 150, "chol": 250, "fbs": 1}
    female_patient = {"age": 30, "sex": 0, "trestbps": 110, "chol": 160, "fbs": 0}

    m_risk = calculate_framingham_risk(male_patient)
    f_risk = calculate_framingham_risk(female_patient)

    assert 0.0 < m_risk <= 1.0
    assert 0.0 < f_risk <= 1.0
    assert m_risk > f_risk, "High-risk male should have higher Framingham score than young healthy female"


def test_threshold_optimizer_minimizes_cost():
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.array([0.05] * 45 + [0.45] * 5 + [0.48] * 15 + [0.85] * 35)

    res = find_optimal_threshold(y_true, y_prob, fn_cost=5.0, fp_cost=1.0)
    assert "optimal_threshold" in res
    assert "cost_reduction_pct" in res
    assert res["optimal_cost"] < res["naive_50_cost"], "Optimal cost must be strictly less than naive cost on borderline cohort"


def test_counterfactual_simulation_reduces_risk():
    model, _ = load_best_model()
    preprocessor = load_preprocessor()

    test_patient = {
        "age": 67, "sex": 1, "cp": 0, "trestbps": 168, "chol": 298,
        "fbs": 1, "restecg": 2, "thalach": 112, "exang": 1, "oldpeak": 3.4,
        "slope": 1, "ca": 2, "thal": 3
    }

    adjustments = {"trestbps": 120, "chol": 195, "fbs": 0, "thalach": 165, "exang": 0, "oldpeak": 0.2, "slope": 2}
    sim_res = simulate_intervention(test_patient, adjustments, model, preprocessor)

    assert "new_probability" in sim_res
    assert "initial_probability" in sim_res
    assert "absolute_reduction" in sim_res
    assert 0.0 <= sim_res["initial_probability"] <= 1.0
    assert 0.0 <= sim_res["new_probability"] <= 1.0
    assert sim_res["new_probability"] < sim_res["initial_probability"], "Clinical intervention must reduce predicted cardiac risk"
    assert np.isclose(
        sim_res["absolute_reduction"],
        sim_res["initial_probability"] - sim_res["new_probability"]
    )


def test_actionable_plans_monotonic_reduction():
    model, _ = load_best_model()
    preprocessor = load_preprocessor()

    patient = {
        "age": 67, "sex": 1, "cp": 0, "trestbps": 168, "chol": 298,
        "fbs": 1, "restecg": 2, "thalach": 112, "exang": 1, "oldpeak": 3.4,
        "slope": 1, "ca": 2, "thal": 3
    }

    init_prob = predict_patient_risk(patient, model, preprocessor)
    plans = generate_actionable_plans(patient, model, preprocessor, target_threshold=0.30)
    assert len(plans) == 3
    # Verify monotonic risk reduction across Tiers 1 -> 2 -> 3
    assert plans[0]["new_probability"] <= init_prob
    assert plans[1]["new_probability"] <= plans[0]["new_probability"]
    assert plans[2]["new_probability"] <= plans[1]["new_probability"]
    assert plans[2]["achieves_target"] is True
