"""
Explainable AI (XAI) Module for CardioCare AI
Implements SHAP (Shapley Additive exPlanations) for global feature importance
and patient-specific local waterfall plots.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import sys
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (
    FEATURE_COLUMNS,
    FEATURE_METADATA,
    load_heart_data,
)
from src.preprocess import load_preprocessor, transform_single_patient
from src.train import load_best_model


class CardioCareExplainer:
    """
    Manages SHAP explainability for the CardioCare AI system.
    Supports both tree-based models (Random Forest, Gradient Boosting)
    and fallback KernelExplainer.
    """

    def __init__(self, model=None, preprocessor=None):
        self.model = model
        self.preprocessor = preprocessor

        if self.model is None or self.preprocessor is None:
            self.model, self.model_name = load_best_model()
            self.preprocessor = load_preprocessor()

        # Load background data for explainer
        df = load_heart_data()
        # Crucial fix: align feature_names with the exact output column order of ColumnTransformer
        if hasattr(self.preprocessor, "get_feature_names_out"):
            self.feature_names = list(self.preprocessor.get_feature_names_out())
        else:
            self.feature_names = FEATURE_COLUMNS

        self.background_raw = df[FEATURE_COLUMNS].sample(n=min(100, len(df)), random_state=42)
        self.background_scaled = self.preprocessor.transform(self.background_raw)

        # Initialize TreeExplainer
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            self.explainer = shap.Explainer(self.model.predict_proba, self.background_scaled)

    def explain_patient(self, patient_dict: Dict) -> Dict:
        """
        Computes SHAP values for a single patient and generates human-readable
        clinical risk attribution aligned with ColumnTransformer output order.
        """
        patient_scaled = transform_single_patient(patient_dict, self.preprocessor)
        shap_res = self.explainer(patient_scaled)

        if len(shap_res.values.shape) == 3:
            values = shap_res.values[0, :, 1]
            base_val = shap_res.base_values[0, 1] if hasattr(shap_res.base_values, "__getitem__") else float(shap_res.base_values)
        else:
            values = shap_res.values[0]
            base_val = float(shap_res.base_values[0]) if hasattr(shap_res.base_values, "__getitem__") else float(shap_res.base_values)

        impacts = []
        for feat_name, impact_val in zip(self.feature_names, values):
            raw_val = patient_dict.get(feat_name, "N/A")
            meta = FEATURE_METADATA.get(feat_name, {})
            human_name = meta.get("name", feat_name)
            unit = meta.get("unit", "")

            if meta.get("type") == "categorical" and "options" in meta:
                display_val = meta["options"].get(raw_val, str(raw_val))
            else:
                display_val = f"{raw_val} {unit}".strip()

            impacts.append({
                "feature": feat_name,
                "label": human_name,
                "value_display": display_val,
                "impact": float(impact_val),
                "is_risk": bool(impact_val > 0)
            })

        impacts_sorted = sorted(impacts, key=lambda x: abs(x["impact"]), reverse=True)

        top_risks = [x for x in impacts_sorted if x["is_risk"]]
        top_protect = [x for x in impacts_sorted if not x["is_risk"]]

        bullets = []
        for item in top_risks[:3]:
            bullets.append(
                f"⚠️ **{item['label']}** ({item['value_display']}): "
                f"Increased predicted cardiac risk by **+{item['impact']:.3f}**."
            )
        for item in top_protect[:3]:
            bullets.append(
                f"🛡️ **{item['label']}** ({item['value_display']}): "
                f"Reduced predicted cardiac risk by **{item['impact']:.3f}**."
            )

        return {
            "shap_values": values,
            "base_value": base_val,
            "patient_scaled": patient_scaled,
            "impacts_sorted": impacts_sorted,
            "top_risks": top_risks,
            "top_protective": top_protect,
            "summary_bullets": bullets
        }

    def generate_waterfall_plot(self, patient_dict: Dict) -> plt.Figure:
        """
        Renders an individual patient SHAP waterfall plot showing positive (red)
        and negative (green) forces acting on the prediction.
        """
        explanation = self.explain_patient(patient_dict)
        impacts = explanation["impacts_sorted"][:10]
        impacts = impacts[::-1]

        labels = [f"{x['label']}\n({x['value_display']})" for x in impacts]
        values = [x["impact"] for x in impacts]
        colors = ["#e63946" if v > 0 else "#2a9d8f" for v in values]

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        y_pos = np.arange(len(labels))

        bars = ax.barh(y_pos, values, color=colors, height=0.6, edgecolor="none", alpha=0.9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.7)

        for bar, val in zip(bars, values):
            width = bar.get_width()
            align = "left" if width >= 0 else "right"
            offset = 0.01 if width >= 0 else -0.01
            sign = "+" if width > 0 else ""
            ax.text(
                width + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{sign}{width:.3f}",
                va="center",
                ha=align,
                fontsize=8.5,
                fontweight="bold",
                color="#e63946" if width > 0 else "#2a9d8f"
            )

        ax.set_xlabel("SHAP Impact on Heart Disease Log-Odds (+ Increases Risk, - Decreases Risk)", fontsize=9.5)
        ax.set_title("Patient-Specific Risk Attribution (Local SHAP Waterfall)", fontsize=11, fontweight="bold", pad=12)
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        plt.tight_layout()

        return fig

    def generate_global_importance_plot(self) -> plt.Figure:
        """
        Computes mean absolute SHAP value across the background dataset
        to display global feature importance ranking.
        """
        shap_res = self.explainer(self.background_scaled)
        if len(shap_res.values.shape) == 3:
            vals = np.abs(shap_res.values[:, :, 1]).mean(axis=0)
        else:
            vals = np.abs(shap_res.values).mean(axis=0)

        labels = [FEATURE_METADATA[f]["name"] for f in self.feature_names]
        sorted_idx = np.argsort(vals)

        sorted_labels = [labels[i] for i in sorted_idx]
        sorted_vals = [vals[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        y_pos = np.arange(len(sorted_labels))

        ax.barh(y_pos, sorted_vals, color="#457b9d", height=0.6, alpha=0.9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_labels, fontsize=9)
        ax.set_xlabel("Mean Absolute SHAP Value (Population Importance)", fontsize=9.5)
        ax.set_title("Global Feature Importance (What drives heart disease overall?)", fontsize=11, fontweight="bold", pad=12)
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        plt.tight_layout()

        return fig


if __name__ == "__main__":
    print("Testing CardioCareExplainer...")
    explainer = CardioCareExplainer()
    sample_patient = {
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
        "ca": 3,
        "thal": 3
    }
    exp = explainer.explain_patient(sample_patient)
    print("\nSummary Bullets for Patient:")
    for b in exp["summary_bullets"]:
        print(" ", b)
