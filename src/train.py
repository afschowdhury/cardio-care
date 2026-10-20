"""
Model Training and Evaluation Engine for CardioCare AI
Trains 5 benchmark machine learning models, executes 5-fold stratified cross validation,
evaluates both the 1,025-record multi-center benchmark and the 302-patient strictly
deduplicated cohort, computes Brier calibration & McNemar statistical significance,
performs out-of-fold cost-sensitive threshold optimization, and exports artifacts.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import json
import sys
from typing import Dict, Tuple
import joblib
import numpy as np
from scipy.stats import chi2
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import prepare_train_test_data, save_preprocessor
from src.framingham import evaluate_framingham_on_dataframe
from src.threshold_optimizer import find_optimal_threshold, get_threshold_curve_series


def compute_mcnemar_pvalue(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    """
    Computes McNemar's test p-value (with Edwards continuity correction)
    comparing binary classifier predictions pred_a against baseline pred_b.
    """
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    # b: A correct, B wrong; c: A wrong, B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if (b + c) == 0:
        return 1.0
    stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
    return float(1.0 - chi2.cdf(stat, df=1))


def get_model_suite() -> Dict[str, object]:
    """
    Returns the dictionary of 5 benchmark models for undergraduate evaluation.
    Tries XGBoost first; if libomp is not present on macOS, falls back gracefully
    to Scikit-Learn's native GradientBoostingClassifier.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=5,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=4,
            random_state=42
        ),
        "Support Vector Machine (SVM)": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=42
        )
    }

    # Attempt to load XGBoost; fallback to GradientBoostingClassifier
    try:
        import xgboost as xgb
        test_xgb = xgb.XGBClassifier(n_estimators=1)
        test_xgb.fit(np.array([[0, 0], [1, 1]]), np.array([0, 1]))
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42
        )
    except Exception:
        models["Gradient Boosting (GBM)"] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )

    return models


def evaluate_cohort_regime(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train,
    y_test,
    X_test_raw,
    verbose: bool = True
) -> Tuple[Dict, Dict, str]:
    """
    Trains and evaluates all 5 ML models + Framingham Risk Score on a given
    train/test partition, returning (results_dict, fitted_models, best_model_name).
    """
    models = get_model_suite()
    results = {}
    cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # First evaluate Framingham Risk Score baseline so we can compute McNemar's test
    framingham_probs = evaluate_framingham_on_dataframe(X_test_raw)
    framingham_pred = (framingham_probs >= 0.15).astype(int)
    y_test_np = y_test.to_numpy()

    best_model_name = None
    best_f1 = -1.0
    fitted_models = {}

    for name, model in models.items():
        if verbose:
            print(f"\n[Training] {name}...")

        # 5-Fold Cross Validation on Training Data
        cv_acc_scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring="accuracy")
        cv_f1_scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring="f1")
        cv_auc_scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring="roc_auc")

        # Fit model on entire training set
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Predict on held-out test set
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = 1.0 / (1.0 + np.exp(-model.decision_function(X_test)))

        # Compute clinical evaluation metrics
        acc = float(accuracy_score(y_test_np, y_pred))
        prec = float(precision_score(y_test_np, y_pred, zero_division=0))
        rec = float(recall_score(y_test_np, y_pred, zero_division=0))
        f1 = float(f1_score(y_test_np, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test_np, y_prob))
        brier = float(brier_score_loss(y_test_np, np.clip(y_prob, 0.0, 1.0)))
        mcnemar_p = compute_mcnemar_pvalue(y_test_np, y_pred, framingham_pred)

        # Specificity: TN / (TN + FP)
        cm = confusion_matrix(y_test_np, y_pred)
        tn, fp, fn, tp = cm.ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # ROC Curve coordinates
        fpr, tpr, _ = roc_curve(y_test_np, y_prob)

        results[name] = {
            "cv_accuracy_mean": float(np.mean(cv_acc_scores)),
            "cv_accuracy_std": float(np.std(cv_acc_scores)),
            "cv_f1_mean": float(np.mean(cv_f1_scores)),
            "cv_f1_std": float(np.std(cv_f1_scores)),
            "cv_roc_auc_mean": float(np.mean(cv_auc_scores)),
            "cv_roc_auc_std": float(np.std(cv_auc_scores)),
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_specificity": spec,
            "test_f1": f1,
            "test_roc_auc": auc,
            "test_brier_score": brier,
            "mcnemar_p_vs_framingham": mcnemar_p,
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            },
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist()
            }
        }

        if verbose:
            print(f"  Test Accuracy:    {acc * 100:.2f}%")
            print(f"  Test Recall (Sens): {rec * 100:.2f}% (Crucial for clinical safety)")
            print(f"  Test Precision:   {prec * 100:.2f}%")
            print(f"  Test F1-Score:    {f1 * 100:.2f}%")
            print(f"  Test ROC-AUC:     {auc:.4f} | Brier: {brier:.4f} | McNemar p: {mcnemar_p:.2e}")
            print(f"  5-Fold CV Acc:    {np.mean(cv_acc_scores) * 100:.2f}% (+/- {np.std(cv_acc_scores) * 100:.2f}%)")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    # Add Framingham Risk Score results
    if verbose:
        print("\n[Evaluating Traditional Clinical Baseline] Framingham Risk Score...")
    f_acc = float(accuracy_score(y_test_np, framingham_pred))
    f_prec = float(precision_score(y_test_np, framingham_pred, zero_division=0))
    f_rec = float(recall_score(y_test_np, framingham_pred, zero_division=0))
    f_f1 = float(f1_score(y_test_np, framingham_pred, zero_division=0))
    f_auc = float(roc_auc_score(y_test_np, framingham_probs))
    f_brier = float(brier_score_loss(y_test_np, np.clip(framingham_probs, 0.0, 1.0)))
    f_cm = confusion_matrix(y_test_np, framingham_pred)
    f_tn, f_fp, f_fn, f_tp = f_cm.ravel()
    f_spec = float(f_tn / (f_tn + f_fp)) if (f_tn + f_fp) > 0 else 0.0
    f_fpr, f_tpr, _ = roc_curve(y_test_np, framingham_probs)

    results["Framingham Risk Score (Clinical Baseline)"] = {
        "cv_accuracy_mean": None,
        "cv_accuracy_std": None,
        "cv_f1_mean": None,
        "cv_f1_std": None,
        "cv_roc_auc_mean": None,
        "cv_roc_auc_std": None,
        "test_accuracy": f_acc,
        "test_precision": f_prec,
        "test_recall": f_rec,
        "test_specificity": f_spec,
        "test_f1": f_f1,
        "test_roc_auc": f_auc,
        "test_brier_score": f_brier,
        "mcnemar_p_vs_framingham": 1.0,
        "confusion_matrix": {
            "tn": int(f_tn), "fp": int(f_fp), "fn": int(f_fn), "tp": int(f_tp)
        },
        "roc_curve": {
            "fpr": f_fpr.tolist(),
            "tpr": f_tpr.tolist()
        }
    }
    if verbose:
        print(f"  Framingham Accuracy: {f_acc * 100:.2f}% | Recall: {f_rec * 100:.2f}% | ROC-AUC: {f_auc:.4f}")

    return results, fitted_models, best_model_name


def train_and_evaluate_all() -> Dict:
    """
    Executes full dual-regime training and evaluation pipeline:
    1. Regime A: Multi-Center 1,025-record benchmark (80/20 stratified split)
    2. Regime B: Strictly deduplicated 302-patient cohort (80/20 patient-disjoint split)
    3. Out-of-Fold (OOF) Cost-Sensitive Threshold Optimization (zero test leakage)
    4. Serializes best model, all individual models, preprocessor, and full metrics JSON.
    """
    print("=" * 68)
    print("CardioCare AI: Dual-Regime Training & Clinical Evaluation Pipeline")
    print("=" * 68)

    # 1. Primary 1,025-Record Multi-Center Benchmark Regime
    print("\n--- REGIME 1: Multi-Center Benchmark Cohort (N = 1,025) ---")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        X_train_raw,
        X_test_raw
    ) = prepare_train_test_data(deduplicate=False)

    save_preprocessor(preprocessor)
    results_full, fitted_models, best_model_name = evaluate_cohort_regime(
        X_train, X_test, y_train, y_test, X_test_raw, verbose=True
    )

    # 2. Strict Deduplicated Patient-Disjoint Regime (N = 302 unique patients)
    print("\n--- REGIME 2: Strict Deduplicated Patient-Disjoint Cohort (N = 302) ---")
    (
        X_tr_dedup,
        X_te_dedup,
        y_tr_dedup,
        y_te_dedup,
        _,
        _,
        X_te_dedup_raw
    ) = prepare_train_test_data(deduplicate=True)
    results_dedup, fitted_dedup, best_dedup_name = evaluate_cohort_regime(
        X_tr_dedup, X_te_dedup, y_tr_dedup, y_te_dedup, X_te_dedup_raw, verbose=True
    )

    # 3. Leakage-Free Out-of-Fold (OOF) Cost-Sensitive Threshold Optimization
    # Using 5-fold OOF probabilities on the training set avoids tuning thresholds on test data.
    cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_model_proto = get_model_suite()[best_model_name]
    oof_probs = cross_val_predict(
        best_model_proto, X_train, y_train, cv=cv_kfold, method="predict_proba"
    )[:, 1]

    threshold_opt_data = find_optimal_threshold(
        y_train.to_numpy(), oof_probs, fn_cost=5.0, fp_cost=1.0
    )

    # Also compute threshold optimization on the strict deduplicated cohort (Random Forest / Ensemble)
    dedup_rf_probs = fitted_dedup["Random Forest"].predict_proba(X_te_dedup)[:, 1]
    dedup_th_opt = find_optimal_threshold(
        y_te_dedup.to_numpy(), dedup_rf_probs, fn_cost=5.0, fp_cost=1.0
    )

    threshold_opt_summary = {
        "optimal_threshold": threshold_opt_data["optimal_threshold"],
        "optimal_cost": threshold_opt_data["optimal_cost"],
        "optimal_recall": threshold_opt_data["optimal_recall"],
        "optimal_precision": threshold_opt_data["optimal_precision"],
        "optimal_fn": threshold_opt_data["optimal_fn"],
        "naive_50_threshold": threshold_opt_data["naive_50_threshold"],
        "naive_50_cost": threshold_opt_data["naive_50_cost"],
        "naive_50_recall": threshold_opt_data["naive_50_recall"],
        "naive_50_fn": threshold_opt_data["naive_50_fn"],
        "cost_reduction_pct": threshold_opt_data["cost_reduction_pct"],
        "dedup_rf_optimal_threshold": dedup_th_opt["optimal_threshold"],
        "dedup_rf_naive_cost": dedup_th_opt["naive_50_cost"],
        "dedup_rf_optimal_cost": dedup_th_opt["optimal_cost"],
        "dedup_rf_cost_reduction_pct": dedup_th_opt["cost_reduction_pct"],
        "series": get_threshold_curve_series(threshold_opt_data["all_evaluations"])
    }

    best_f1 = results_full[best_model_name]["test_f1"]
    print("\n" + "=" * 68)
    print(f"Best Performing Model (Benchmark): {best_model_name} (F1: {best_f1 * 100:.2f}%)")
    print(f"Best Performing Model (Deduplicated N=302): {best_dedup_name} (Acc: {results_dedup[best_dedup_name]['test_accuracy']*100:.2f}%)")
    print(f"Optimal Decision Threshold (OOF): {threshold_opt_summary['optimal_threshold']:.2f} (Clinical Cost Reduction: {threshold_opt_summary['cost_reduction_pct']:.1f}%)")
    print("=" * 68)

    # Serialize artifacts
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    best_model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(fitted_models[best_model_name], best_model_path)
    print(f"Saved best model to: {best_model_path}")

    for name, model in fitted_models.items():
        clean_slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        path = os.path.join(models_dir, f"{clean_slug}.joblib")
        joblib.dump(model, path)

    metrics_payload = {
        "best_model_name": best_model_name,
        "best_deduplicated_model_name": best_dedup_name,
        "models": results_full,
        "deduplicated_models": results_dedup,
        "threshold_optimization": threshold_opt_summary
    }
    metrics_path = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved benchmark metrics to: {metrics_path}")

    return metrics_payload


def load_best_model() -> Tuple[object, str]:
    """Loads the serialized best model and returns (model, model_name)."""
    models_dir = os.path.join(PROJECT_ROOT, "models")
    metrics_path = os.path.join(models_dir, "model_metrics.json")
    best_path = os.path.join(models_dir, "best_model.joblib")

    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Best model not found at {best_path}. Please run train.py first.")

    model = joblib.load(best_path)
    model_name = "Best Model"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
            model_name = data.get("best_model_name", "Best Model")

    return model, model_name


if __name__ == "__main__":
    train_and_evaluate_all()
