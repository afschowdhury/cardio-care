"""
Cost-Sensitive Clinical Threshold Optimizer for CardioCare AI
Eliminates dangerous False Negatives (Type II clinical errors) by optimizing
the decision threshold using an asymmetric medical cost matrix.
"""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix, fbeta_score, precision_recall_curve


def compute_clinical_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0
) -> Dict[str, float]:
    """
    Computes diagnostic metrics and total clinical penalty cost at a given threshold.
    In clinical medicine, FN (missed coronary stenosis) carries a much higher penalty
    than FP (unnecessary confirmatory ECG / blood test).
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    total_cost = (fn * fn_cost) + (fp * fp_cost)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)

    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "total_cost": float(total_cost),
        "recall": float(recall),
        "precision": float(precision),
        "f2_score": float(f2)
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0
) -> Dict:
    """
    Scans decision thresholds from 0.05 to 0.95 to find the threshold
    that minimizes the total clinical cost.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    evaluations = [
        compute_clinical_cost(y_true, y_prob, t, fn_cost=fn_cost, fp_cost=fp_cost)
        for t in thresholds
    ]

    # Find minimum cost entry
    best_eval = min(evaluations, key=lambda x: x["total_cost"])

    # Baseline cost at naive 0.50 threshold
    naive_eval = compute_clinical_cost(y_true, y_prob, 0.50, fn_cost=fn_cost, fp_cost=fp_cost)

    cost_reduction_pct = (
        (naive_eval["total_cost"] - best_eval["total_cost"]) / naive_eval["total_cost"] * 100
        if naive_eval["total_cost"] > 0 else 0.0
    )

    return {
        "optimal_threshold": best_eval["threshold"],
        "optimal_cost": best_eval["total_cost"],
        "optimal_recall": best_eval["recall"],
        "optimal_precision": best_eval["precision"],
        "optimal_fn": best_eval["fn"],
        "naive_50_threshold": 0.50,
        "naive_50_cost": naive_eval["total_cost"],
        "naive_50_recall": naive_eval["recall"],
        "naive_50_fn": naive_eval["fn"],
        "cost_reduction_pct": cost_reduction_pct,
        "all_evaluations": evaluations
    }


def get_threshold_curve_series(evaluations: List[Dict]) -> Dict[str, List[float]]:
    """Extracts series coordinates for Streamlit / Matplotlib plotting."""
    return {
        "thresholds": [e["threshold"] for e in evaluations],
        "costs": [e["total_cost"] for e in evaluations],
        "recalls": [e["recall"] for e in evaluations],
        "precisions": [e["precision"] for e in evaluations],
        "f2_scores": [e["f2_score"] for e in evaluations],
        "false_negatives": [e["fn"] for e in evaluations]
    }


if __name__ == "__main__":
    # Test on synthetic data
    y_t = np.array([0] * 50 + [1] * 50)
    y_p = np.array([0.1] * 45 + [0.45] * 5 + [0.48] * 10 + [0.85] * 40)
    res = find_optimal_threshold(y_t, y_p, fn_cost=5.0, fp_cost=1.0)
    print(f"Optimal Threshold: {res['optimal_threshold']:.2f}")
    print(f"Naive 0.50 Cost:   {res['naive_50_cost']} (FN: {res['naive_50_fn']})")
    print(f"Optimal Cost:      {res['optimal_cost']} (FN: {res['optimal_fn']})")
    print(f"Cost Reduction:    {res['cost_reduction_pct']:.1f}%")
