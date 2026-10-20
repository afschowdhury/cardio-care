"""
Framingham Risk Score Module for CardioCare AI
Implements the clinical gold-standard Framingham 10-year cardiovascular risk
scoring algorithm (Wilson et al., Circulation / D'Agostino et al., 2008).
Used as the primary medical baseline to compare against Machine Learning models.
"""

from typing import Dict, Union
import numpy as np
import pandas as pd


def calculate_framingham_risk(patient: Union[Dict, pd.Series]) -> float:
    """
    Calculates the 10-year cardiovascular risk probability according to the
    standardized adult Framingham point system.
    
    Inputs used from dataset:
      - age (years)
      - sex (1: Male, 0: Female)
      - trestbps (Resting Systolic Blood Pressure, mm Hg)
      - chol (Total Serum Cholesterol, mg/dl)
      - fbs (Fasting blood sugar > 120 mg/dl, indicator of diabetes)
      
    Returns:
      float: 10-year CVD risk probability between 0.01 and 0.95.
    """
    age = float(patient["age"])
    sex = int(patient["sex"])  # 1 = Male, 0 = Female
    sbp = float(patient["trestbps"])
    chol = float(patient["chol"])
    diabetes = int(patient.get("fbs", 0))

    points = 0

    if sex == 1:
        # ------------------ MEN'S POINT SYSTEM ------------------
        # Age
        if age < 35:
            points -= 9
        elif age <= 39:
            points -= 4
        elif age <= 44:
            points += 0
        elif age <= 49:
            points += 3
        elif age <= 54:
            points += 6
        elif age <= 59:
            points += 8
        elif age <= 64:
            points += 10
        elif age <= 69:
            points += 11
        elif age <= 74:
            points += 12
        else:
            points += 13

        # Total Cholesterol (Age-adjusted)
        if age < 40:
            if chol < 160: points += 0
            elif chol <= 199: points += 4
            elif chol <= 239: points += 7
            elif chol <= 279: points += 9
            else: points += 11
        elif age <= 49:
            if chol < 160: points += 0
            elif chol <= 199: points += 3
            elif chol <= 239: points += 5
            elif chol <= 279: points += 6
            else: points += 8
        elif age <= 59:
            if chol < 160: points += 0
            elif chol <= 199: points += 2
            elif chol <= 239: points += 3
            elif chol <= 279: points += 4
            else: points += 5
        elif age <= 69:
            if chol < 160: points += 0
            elif chol <= 199: points += 1
            elif chol <= 239: points += 1
            elif chol <= 279: points += 2
            else: points += 3
        else:
            if chol < 160: points += 0
            elif chol <= 199: points += 0
            elif chol <= 239: points += 0
            elif chol <= 279: points += 1
            else: points += 1

        # Systolic Blood Pressure
        if sbp < 120: points += 0
        elif sbp <= 129: points += 0
        elif sbp <= 139: points += 1
        elif sbp <= 159: points += 2
        else: points += 3

        # Diabetes / Elevated Fasting Glucose
        if diabetes == 1:
            points += 2

        # Convert Men's points to risk probability (Wilson et al.)
        if points <= -3: risk = 0.01
        elif points <= 0: risk = 0.02
        elif points == 1: risk = 0.02
        elif points == 2: risk = 0.03
        elif points == 3: risk = 0.03
        elif points == 4: risk = 0.04
        elif points == 5: risk = 0.04
        elif points == 6: risk = 0.05
        elif points == 7: risk = 0.06
        elif points == 8: risk = 0.07
        elif points == 9: risk = 0.08
        elif points == 10: risk = 0.10
        elif points == 11: risk = 0.11
        elif points == 12: risk = 0.13
        elif points == 13: risk = 0.15
        elif points == 14: risk = 0.18
        elif points == 15: risk = 0.20
        elif points == 16: risk = 0.24
        elif points == 17: risk = 0.27
        elif points == 18: risk = 0.30
        else: risk = min(0.35 + (points - 18) * 0.03, 0.95)

    else:
        # ------------------ WOMEN'S POINT SYSTEM ------------------
        # Age
        if age < 35:
            points -= 7
        elif age <= 39:
            points -= 3
        elif age <= 44:
            points += 0
        elif age <= 49:
            points += 3
        elif age <= 54:
            points += 6
        elif age <= 59:
            points += 8
        elif age <= 64:
            points += 10
        elif age <= 69:
            points += 12
        elif age <= 74:
            points += 14
        else:
            points += 16

        # Total Cholesterol (Age-adjusted)
        if age < 40:
            if chol < 160: points += 0
            elif chol <= 199: points += 4
            elif chol <= 239: points += 8
            elif chol <= 279: points += 11
            else: points += 13
        elif age <= 49:
            if chol < 160: points += 0
            elif chol <= 199: points += 3
            elif chol <= 239: points += 6
            elif chol <= 279: points += 8
            else: points += 10
        elif age <= 59:
            if chol < 160: points += 0
            elif chol <= 199: points += 2
            elif chol <= 239: points += 4
            elif chol <= 279: points += 5
            else: points += 7
        elif age <= 69:
            if chol < 160: points += 0
            elif chol <= 199: points += 1
            elif chol <= 239: points += 2
            elif chol <= 279: points += 3
            else: points += 4
        else:
            if chol < 160: points += 0
            elif chol <= 199: points += 1
            elif chol <= 239: points += 1
            elif chol <= 279: points += 2
            else: points += 2

        # Systolic Blood Pressure
        if sbp < 120: points += 0
        elif sbp <= 129: points += 1
        elif sbp <= 139: points += 2
        elif sbp <= 159: points += 3
        else: points += 4

        # Diabetes / Elevated Fasting Glucose
        if diabetes == 1:
            points += 4

        # Convert Women's points to risk probability
        if points <= -2: risk = 0.01
        elif points <= 2: risk = 0.01
        elif points == 3: risk = 0.02
        elif points == 4: risk = 0.02
        elif points == 5: risk = 0.03
        elif points == 6: risk = 0.03
        elif points == 7: risk = 0.04
        elif points == 8: risk = 0.04
        elif points == 9: risk = 0.05
        elif points == 10: risk = 0.06
        elif points == 11: risk = 0.07
        elif points == 12: risk = 0.08
        elif points == 13: risk = 0.10
        elif points == 14: risk = 0.11
        elif points == 15: risk = 0.13
        elif points == 16: risk = 0.15
        elif points == 17: risk = 0.18
        elif points == 18: risk = 0.20
        elif points == 19: risk = 0.24
        elif points == 20: risk = 0.27
        elif points == 21: risk = 0.30
        else: risk = min(0.35 + (points - 21) * 0.03, 0.95)

    return float(np.clip(risk, 0.01, 0.95))


def evaluate_framingham_on_dataframe(df: pd.DataFrame) -> np.ndarray:
    """
    Applies the Framingham risk calculator across every patient row in a DataFrame.
    Returns 1D array of risk probabilities.
    """
    return np.array([calculate_framingham_risk(row) for _, row in df.iterrows()])


if __name__ == "__main__":
    sample_m = {"age": 65, "sex": 1, "trestbps": 160, "chol": 260, "fbs": 1}
    sample_f = {"age": 35, "sex": 0, "trestbps": 115, "chol": 170, "fbs": 0}
    print(f"Sample Male (65y, BP 160, Chol 260, Diabetic): {calculate_framingham_risk(sample_m) * 100:.1f}% risk")
    print(f"Sample Female (35y, BP 115, Chol 170, Normal):   {calculate_framingham_risk(sample_f) * 100:.1f}% risk")
