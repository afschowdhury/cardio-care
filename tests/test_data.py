"""
Unit Tests for Data Ingestion, Canonical Target Polarity, Deduplication, and Preprocessing Pipeline
"""

import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_heart_data,
)
from src.preprocess import (
    prepare_train_test_data,
    transform_single_patient,
)


def test_load_heart_data():
    df = load_heart_data()
    assert df is not None
    assert len(df) == 1025
    assert len(df.columns) == 14
    assert TARGET_COLUMN in df.columns
    for col in FEATURE_COLUMNS:
        assert col in df.columns


def test_deduplicated_cohort_loading():
    df_dedup = load_heart_data(deduplicate=True)
    assert len(df_dedup) == 302
    assert df_dedup.duplicated().sum() == 0


def test_canonical_target_physiological_polarity():
    df = load_heart_data(deduplicate=True)
    # Patients with coronary disease (target=1) must have higher mean ST depression and vessel count
    mean_oldpeak_healthy = df.loc[df[TARGET_COLUMN] == 0, "oldpeak"].mean()
    mean_oldpeak_disease = df.loc[df[TARGET_COLUMN] == 1, "oldpeak"].mean()
    mean_ca_healthy = df.loc[df[TARGET_COLUMN] == 0, "ca"].mean()
    mean_ca_disease = df.loc[df[TARGET_COLUMN] == 1, "ca"].mean()
    assert mean_oldpeak_disease > mean_oldpeak_healthy
    assert mean_ca_disease > mean_ca_healthy


def test_no_null_values():
    df = load_heart_data()
    assert df.isnull().sum().sum() == 0, "Dataset contains unexpected null values"


def test_target_values():
    df = load_heart_data()
    unique_targets = set(df[TARGET_COLUMN].unique())
    assert unique_targets == {0, 1}, f"Target should be binary (0, 1), got {unique_targets}"


def test_prepare_train_test_split():
    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        X_train_raw,
        X_test_raw
    ) = prepare_train_test_data(test_size=0.2, random_state=42)

    assert X_train.shape == (820, 13)
    assert X_test.shape == (205, 13)
    assert len(y_train) == 820
    assert len(y_test) == 205
    assert preprocessor is not None


def test_single_patient_transform():
    _, _, _, _, preprocessor, _, _ = prepare_train_test_data()
    sample_patient = {
        "age": 55,
        "sex": 1,
        "cp": 0,
        "trestbps": 130,
        "chol": 240,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 1,
        "ca": 0,
        "thal": 2
    }
    transformed = transform_single_patient(sample_patient, preprocessor)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == (1, 13)
