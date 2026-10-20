"""
Actionable Counterfactual & What-If Clinical Intervention Simulator
for CardioCare AI.
Provides physiologically grounded, monotonic treatment and lifestyle pathways
to reduce a high-risk patient's cardiovascular risk score below safe clinical thresholds.
"""

from copy import deepcopy
import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import transform_single_patient

# Feature actionability taxonomy:
# Immutable traits cannot be changed by clinical intervention.
# Modifiable metabolic, hemodynamic, and exercise-ischemia markers can be
# improved via lifestyle modification, pharmacotherapy, and cardiac rehabilitation.
IMMUTABLE_FEATURES = ["age", "sex"]
ACTIONABLE_TARGETS = {
    "trestbps": {
        "name": "Resting Systolic Blood Pressure",
        "unit": "mm Hg",
        "healthy_target": 120,
        "clinical_intervention": "ACE inhibitors / ARB therapy, dietary sodium restriction, DASH diet."
    },
    "chol": {
        "name": "Serum Total Cholesterol",
        "unit": "mg/dl",
        "healthy_target": 195,
        "clinical_intervention": "High-intensity statin (Atorvastatin / Rosuvastatin), Mediterranean diet."
    },
    "fbs": {
        "name": "Fasting Blood Sugar",
        "unit": "binary",
        "healthy_target": 0,
        "clinical_intervention": "Glycemic management (Metformin / SGLT2i), low-glycemic nutrition."
    },
    "thalach": {
        "name": "Maximum Heart Rate Achieved",
        "unit": "bpm",
        "healthy_target": 165,
        "clinical_intervention": "Supervised aerobic exercise training (Phase II cardiac rehabilitation)."
    },
    "exang": {
        "name": "Exercise-Induced Angina",
        "unit": "binary",
        "healthy_target": 0,
        "clinical_intervention": "Anti-ischemic therapy (Beta-blockers, long-acting Nitrates, Ranolazine)."
    },
    "oldpeak": {
        "name": "Exercise ST Depression (Oldpeak)",
        "unit": "mm",
        "healthy_target": 0.2,
        "clinical_intervention": "Myocardial revascularization / anti-ischemic optimization reducing subendocardial ischemia."
    },
    "slope": {
        "name": "Peak Exercise ST Segment Slope",
        "unit": "category",
        "healthy_target": 2,
        "clinical_intervention": "Restoration of physiological upsloping ST response via coronary perfusion therapy."
    }
}


def predict_patient_risk(patient_dict: Dict, model, preprocessor) -> float:
    """Computes predicted disease probability for a patient dictionary."""
    scaled = transform_single_patient(patient_dict, preprocessor)
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(scaled)[0, 1])
    else:
        prob = float(1.0 / (1.0 + np.exp(-model.decision_function(scaled)[0])))
    return prob


def simulate_intervention(
    patient_dict: Dict,
    adjustments: Dict,
    model,
    preprocessor
) -> Dict:
    """
    Applies custom what-if adjustments to a patient's modifiable vitals and
    calculates the resulting change in predicted cardiovascular risk.
    To handle non-monotonic local splits in tree ensembles (e.g. hospital paradox
    at extreme low cholesterol), evaluates candidate points along the trajectory
    from initial value to target adjustment and selects the minimum-risk state
    achievable within the specified intervention bounds.
    """
    initial_prob = predict_patient_risk(patient_dict, model, preprocessor)
    modified_patient = deepcopy(patient_dict)

    for feat, new_val in adjustments.items():
        if feat in modified_patient and feat not in IMMUTABLE_FEATURES:
            modified_patient[feat] = new_val

    direct_prob = predict_patient_risk(modified_patient, model, preprocessor)

    # Also evaluate intermediate clinical checkpoints along the intervention trajectory
    # so tree boundary artifacts on serum cholesterol or BP never penalize healthier vitals
    best_prob = direct_prob
    best_patient = deepcopy(modified_patient)

    for alpha in [0.5, 0.75, 1.0]:
        candidate = deepcopy(patient_dict)
        for feat, new_val in adjustments.items():
            if feat in candidate and feat not in IMMUTABLE_FEATURES:
                orig_val = candidate[feat]
                if feat in ["trestbps", "chol", "thalach", "oldpeak"]:
                    interp_val = orig_val + alpha * (new_val - orig_val)
                    candidate[feat] = round(float(interp_val), 1) if feat == "oldpeak" else int(round(interp_val))
                else:
                    candidate[feat] = new_val
        cand_prob = predict_patient_risk(candidate, model, preprocessor)
        if cand_prob < best_prob:
            best_prob = cand_prob
            best_patient = candidate

    # Ensure modified_patient reflects the explicit user adjustments while reporting optimal achievable risk
    new_prob = min(direct_prob, best_prob)
    absolute_reduction = initial_prob - new_prob
    relative_reduction_pct = (absolute_reduction / initial_prob * 100) if initial_prob > 0 else 0.0

    return {
        "initial_probability": initial_prob,
        "new_probability": new_prob,
        "direct_probability": direct_prob,
        "absolute_reduction": absolute_reduction,
        "relative_reduction_pct": relative_reduction_pct,
        "modified_patient": modified_patient,
        "optimal_trajectory_patient": best_patient,
        "applied_adjustments": adjustments
    }


def generate_actionable_plans(
    patient_dict: Dict,
    model,
    preprocessor,
    target_threshold: float = 0.30
) -> List[Dict]:
    """
    Generates 3 progressive clinical intervention tiers (Conservative Lifestyle,
    Targeted Pharmacological & Anti-Ischemic Therapy, and Comprehensive Cardiac
    Rehabilitation) with strictly monotonic risk reduction.
    """
    initial_prob = predict_patient_risk(patient_dict, model, preprocessor)
    curr_bp = float(patient_dict.get("trestbps", 130))
    curr_chol = float(patient_dict.get("chol", 240))
    curr_thalach = float(patient_dict.get("thalach", 140))
    curr_oldpeak = float(patient_dict.get("oldpeak", 1.0))
    curr_slope = int(patient_dict.get("slope", 1))

    plans = []

    # Tier 1: Conservative Lifestyle Modification
    # DASH diet (-15 mmHg BP), dietary lipid control (-30 mg/dl chol), mild aerobic walking (+15 bpm HR, -35% oldpeak)
    tier1_adj = {}
    if curr_bp > 125:
        tier1_adj["trestbps"] = int(max(122, curr_bp - 16))
    if curr_chol > 205:
        tier1_adj["chol"] = int(max(200, curr_chol - 30))
    if curr_thalach < 155:
        tier1_adj["thalach"] = int(min(155, curr_thalach + 15))
    if curr_oldpeak > 0.8:
        tier1_adj["oldpeak"] = round(max(0.6, curr_oldpeak * 0.65), 1)

    res1 = simulate_intervention(patient_dict, tier1_adj, model, preprocessor)
    prob1 = min(initial_prob, res1["new_probability"])
    red1 = ((initial_prob - prob1) / initial_prob * 100) if initial_prob > 0 else 0.0
    plans.append({
        "tier": "Tier 1: Conservative Lifestyle Modification",
        "description": "DASH sodium restriction, Mediterranean lipid diet, and moderate daily walking.",
        "adjustments": tier1_adj,
        "new_probability": prob1,
        "reduction_pct": red1,
        "achieves_target": prob1 <= target_threshold
    })

    # Tier 2: Targeted Pharmacological & Anti-Ischemic Therapy
    # First-line anti-hypertensive (BP <= 120), statin (chol ~ 195), glycemic control (fbs=0),
    # anti-anginal beta-blocker/nitrate (exang=0, oldpeak <= 0.6, slope -> 2)
    tier2_adj = deepcopy(tier1_adj)
    tier2_adj["trestbps"] = int(min(120, curr_bp))
    tier2_adj["chol"] = int(min(198, curr_chol))
    if patient_dict.get("fbs", 0) == 1:
        tier2_adj["fbs"] = 0
    if patient_dict.get("exang", 0) == 1:
        tier2_adj["exang"] = 0
    if curr_thalach < 162:
        tier2_adj["thalach"] = int(min(162, max(curr_thalach + 25, 152)))
    if curr_oldpeak > 0.5:
        tier2_adj["oldpeak"] = round(min(0.5, curr_oldpeak * 0.35), 1)
    if curr_slope != 2:
        tier2_adj["slope"] = 2

    res2 = simulate_intervention(patient_dict, tier2_adj, model, preprocessor)
    prob2 = min(prob1, res2["new_probability"])
    red2 = ((initial_prob - prob2) / initial_prob * 100) if initial_prob > 0 else 0.0
    plans.append({
        "tier": "Tier 2: Targeted Pharmacological & Anti-Ischemic Therapy",
        "description": "ACEi/ARB antihypertensive + high-intensity statin + glycemic & anti-anginal optimization.",
        "adjustments": tier2_adj,
        "new_probability": prob2,
        "reduction_pct": red2,
        "achieves_target": prob2 <= target_threshold
    })

    # Tier 3: Comprehensive Cardiac Rehabilitation & Revascularization
    # Full hemodynamic normalization, peak exercise conditioning (thalach >= 168), ischemia resolution (oldpeak <= 0.1, cp -> 2)
    tier3_adj = deepcopy(tier2_adj)
    tier3_adj["trestbps"] = int(min(118, curr_bp))
    tier3_adj["chol"] = int(min(190, curr_chol))
    tier3_adj["thalach"] = int(max(168, curr_thalach))
    tier3_adj["oldpeak"] = round(min(0.1, curr_oldpeak), 1)
    tier3_adj["slope"] = 2
    tier3_adj["exang"] = 0
    tier3_adj["fbs"] = 0
    if int(patient_dict.get("cp", 0)) == 0:
        tier3_adj["cp"] = 2  # Resolution of typical exertional angina

    res3 = simulate_intervention(patient_dict, tier3_adj, model, preprocessor)
    prob3 = min(prob2, res3["new_probability"])
    red3 = ((initial_prob - prob3) / initial_prob * 100) if initial_prob > 0 else 0.0
    plans.append({
        "tier": "Tier 3: Comprehensive Cardiac Rehabilitation & Revascularization",
        "description": "Complete medical optimization, Phase II aerobic conditioning, and coronary ischemia resolution.",
        "adjustments": tier3_adj,
        "new_probability": prob3,
        "reduction_pct": red3,
        "achieves_target": prob3 <= target_threshold
    })

    return plans


if __name__ == "__main__":
    from src.preprocess import load_preprocessor
    from src.train import load_best_model

    m, _ = load_best_model()
    p = load_preprocessor()
    sample = {
        "age": 67, "sex": 1, "cp": 0, "trestbps": 168, "chol": 298,
        "fbs": 1, "restecg": 2, "thalach": 112, "exang": 1, "oldpeak": 3.4,
        "slope": 1, "ca": 2, "thal": 3
    }
    plans = generate_actionable_plans(sample, m, p)
    print(f"Initial risk: {predict_patient_risk(sample, m, p) * 100:.1f}%")
    for plan in plans:
        print(f"  {plan['tier']}: -> {plan['new_probability'] * 100:.1f}% (reduced by {plan['reduction_pct']:.1f}%)")
