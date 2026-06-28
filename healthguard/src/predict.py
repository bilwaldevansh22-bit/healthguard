"""
HealthGuard - Inference Script
------------------------------------
Loads trained models and makes diabetes / heart disease risk
predictions for a new patient record. This simulates how the
model would be used in a real deployment (e.g. behind an API).
"""

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = '/home/claude/projects/healthguard/models'


def load_artifacts():
    encoders = joblib.load(f'{MODEL_DIR}/encoders.pkl')
    feature_cols = joblib.load(f'{MODEL_DIR}/feature_cols.pkl')

    diabetes_model = joblib.load(f'{MODEL_DIR}/best_model_diabetes_risk.pkl')
    diabetes_scaler = joblib.load(f'{MODEL_DIR}/scaler_diabetes_risk.pkl')

    heart_model = joblib.load(f'{MODEL_DIR}/best_model_heart_disease_risk.pkl')
    heart_scaler = joblib.load(f'{MODEL_DIR}/scaler_heart_disease_risk.pkl')

    return {
        'encoders': encoders,
        'feature_cols': feature_cols,
        'diabetes_model': diabetes_model,
        'diabetes_scaler': diabetes_scaler,
        'heart_model': heart_model,
        'heart_scaler': heart_scaler,
    }


def predict_risk(patient_dict, artifacts):
    df = pd.DataFrame([patient_dict])

    for col, encoder in artifacts['encoders'].items():
        df[col] = encoder.transform(df[col])

    X = df[artifacts['feature_cols']]

    X_diabetes = artifacts['diabetes_scaler'].transform(X)
    diabetes_prob = artifacts['diabetes_model'].predict_proba(X_diabetes)[0][1]

    X_heart = artifacts['heart_scaler'].transform(X)
    heart_prob = artifacts['heart_model'].predict_proba(X_heart)[0][1]

    def risk_band(prob):
        if prob < 0.3:
            return 'Low'
        elif prob < 0.6:
            return 'Moderate'
        else:
            return 'High'

    return {
        'diabetes_risk_probability': round(float(diabetes_prob), 3),
        'diabetes_risk_band': risk_band(diabetes_prob),
        'heart_disease_risk_probability': round(float(heart_prob), 3),
        'heart_disease_risk_band': risk_band(heart_prob),
    }


if __name__ == "__main__":
    artifacts = load_artifacts()

    sample_patient = {
        'age': 52,
        'gender': 'Male',
        'bmi': 29.4,
        'systolic_bp': 138,
        'diastolic_bp': 88,
        'glucose_level': 145.0,
        'cholesterol': 230,
        'heart_rate': 78,
        'physical_activity_level': 'Low',
        'smoker': 1,
        'alcohol_consumption': 'Moderate',
        'family_history': 1,
        'sleep_hours': 5.5,
        'stress_level': 8,
    }

    result = predict_risk(sample_patient, artifacts)
    print("Sample Patient Risk Assessment:")
    print(f"  Diabetes Risk:       {result['diabetes_risk_probability']*100:.1f}%  ({result['diabetes_risk_band']})")
    print(f"  Heart Disease Risk:  {result['heart_disease_risk_probability']*100:.1f}%  ({result['heart_disease_risk_band']})")
