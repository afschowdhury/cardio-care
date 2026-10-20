"""
Preprocessing Module for CardioCare AI
Handles feature scaling, transformation, train-test splitting (both standard
multi-center benchmark and strict patient-deduplicated regimes), and artifact serialization.
"""

import os
import sys
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_heart_data,
)

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that applies StandardScaler to numerical
    features and passes categorical integer features through.
    Standard tree ensembles (Random Forest, Gradient Boosting) and linear/kernel
    models benefit significantly from continuous feature normalization.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", "passthrough", CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor


def prepare_train_test_data(
    test_size: float = 0.2,
    random_state: int = 42,
    deduplicate: bool = False
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, ColumnTransformer, pd.DataFrame, pd.DataFrame]:
    """
    Loads data, performs stratified train-test splitting, fits preprocessor strictly
    on the training partition, and returns scaled matrices alongside raw DataFrames.
    
    Args:
        test_size: Proportion of dataset allocated to held-out test partition.
        random_state: Seed for reproducible stratified splitting.
        deduplicate: If True, deduplicates patient records prior to splitting to
                     evaluate strict patient-disjoint generalization (N=302).
    """
    df = load_heart_data(deduplicate=deduplicate)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Stratified split ensures identical class ratio in train and test splits
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train_raw)
    X_test_scaled = preprocessor.transform(X_test_raw)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        preprocessor,
        X_train_raw,
        X_test_raw
    )


def save_preprocessor(preprocessor: ColumnTransformer, output_path: str = None) -> str:
    """Serializes fitted preprocessor to disk."""
    if output_path is None:
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
        )
        os.makedirs(models_dir, exist_ok=True)
        output_path = os.path.join(models_dir, "preprocessor.joblib")

    joblib.dump(preprocessor, output_path)
    return output_path


def load_preprocessor(input_path: str = None) -> ColumnTransformer:
    """Loads serialized preprocessor from disk."""
    if input_path is None:
        input_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "preprocessor.joblib"
        )
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Preprocessor artifact not found at {input_path}")
    return joblib.load(input_path)


def transform_single_patient(patient_dict: Dict, preprocessor: ColumnTransformer) -> np.ndarray:
    """
    Converts a single patient's dictionary of clinical vitals into the
    exact feature matrix format required for model inference.
    """
    df_patient = pd.DataFrame([patient_dict])[FEATURE_COLUMNS]
    return preprocessor.transform(df_patient)


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, prep, X_tr_raw, X_te_raw = prepare_train_test_data()
    saved_path = save_preprocessor(prep)
    print("Data preprocessed successfully:")
    print(f"  X_train shape: {X_tr.shape}, y_train shape: {y_tr.shape}")
    print(f"  X_test shape:  {X_te.shape}, y_test shape:  {y_te.shape}")
    print(f"  Saved preprocessor to: {saved_path}")
