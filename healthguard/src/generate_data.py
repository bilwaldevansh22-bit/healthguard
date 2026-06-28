"""
HealthGuard - Synthetic Patient Health Risk Dataset Generator
------------------------------------------------------------------
Generates a realistic patient dataset for diabetes/heart disease
risk prediction. Feature correlations mimic real clinical patterns
(e.g., BMI-glucose correlation, age-blood pressure correlation,
risk factors compounding realistically) based on patterns commonly
seen in datasets like Pima Indians Diabetes and UCI Heart Disease.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_PATIENTS = 4000


def generate_health_dataset(n=N_PATIENTS):
    age = np.random.randint(18, 80, n)

    # BMI correlates loosely with age and lifestyle
    bmi = np.clip(np.random.normal(26 + age * 0.03, 4.5, n), 15, 50).round(1)

    # Blood pressure rises with age
    systolic_bp = np.clip(np.random.normal(110 + age * 0.5, 12, n), 90, 200).astype(int)
    diastolic_bp = np.clip(systolic_bp * 0.65 + np.random.normal(0, 5, n), 60, 120).astype(int)

    # Glucose correlates with BMI and age (diabetes risk factor)
    glucose = np.clip(
        np.random.normal(90 + bmi * 1.3 + age * 0.25, 18, n), 70, 300
    ).round(1)

    cholesterol = np.clip(np.random.normal(180 + age * 0.6, 30, n), 120, 350).astype(int)

    # Lifestyle factors
    physical_activity_level = np.random.choice(['Low', 'Moderate', 'High'], n, p=[0.4, 0.4, 0.2])
    smoker = np.random.choice([0, 1], n, p=[0.75, 0.25])
    alcohol_consumption = np.random.choice(['None', 'Moderate', 'Heavy'], n, p=[0.5, 0.35, 0.15])
    family_history = np.random.choice([0, 1], n, p=[0.7, 0.3])

    heart_rate = np.clip(np.random.normal(75, 10, n), 50, 120).astype(int)
    sleep_hours = np.clip(np.random.normal(6.8, 1.3, n), 3, 10).round(1)
    stress_level = np.random.randint(1, 11, n)  # self-reported 1-10

    gender = np.random.choice(['Male', 'Female'], n, p=[0.49, 0.51])

    # --- Compute latent risk score (drives both diabetes and heart disease labels) ---
    activity_penalty = np.select(
        [physical_activity_level == 'Low', physical_activity_level == 'Moderate', physical_activity_level == 'High'],
        [1.0, 0.5, 0.0]
    )
    alcohol_penalty = np.select(
        [alcohol_consumption == 'None', alcohol_consumption == 'Moderate', alcohol_consumption == 'Heavy'],
        [0.0, 0.3, 0.8]
    )

    diabetes_risk_score = (
        0.05 * (glucose - 90) +
        0.08 * (bmi - 25) +
        0.03 * (age - 30) +
        1.2 * family_history +
        0.7 * activity_penalty +
        np.random.normal(0, 0.8, n)
    )
    diabetes_prob = 1 / (1 + np.exp(-(diabetes_risk_score - 4.5) / 1.6))
    diabetes = (np.random.rand(n) < diabetes_prob).astype(int)

    heart_risk_score = (
        0.04 * (systolic_bp - 110) +
        0.025 * (cholesterol - 180) +
        0.04 * (age - 30) +
        1.4 * smoker +
        1.0 * family_history +
        0.6 * activity_penalty +
        0.4 * alcohol_penalty +
        0.2 * (stress_level - 5) +
        np.random.normal(0, 0.8, n)
    )
    heart_prob = 1 / (1 + np.exp(-(heart_risk_score - 5.5) / 1.6))
    heart_disease = (np.random.rand(n) < heart_prob).astype(int)

    df = pd.DataFrame({
        'patient_id': [f"PT{i:05d}" for i in range(1, n + 1)],
        'age': age,
        'gender': gender,
        'bmi': bmi,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'glucose_level': glucose,
        'cholesterol': cholesterol,
        'heart_rate': heart_rate,
        'physical_activity_level': physical_activity_level,
        'smoker': smoker,
        'alcohol_consumption': alcohol_consumption,
        'family_history': family_history,
        'sleep_hours': sleep_hours,
        'stress_level': stress_level,
        'diabetes_risk': diabetes,
        'heart_disease_risk': heart_disease,
    })

    return df


if __name__ == "__main__":
    df = generate_health_dataset()
    df.to_csv('/home/claude/projects/healthguard/data/patient_health_data.csv', index=False)
    print(f"Generated {len(df)} patient records")
    print(df.head())
    print(f"\nDiabetes risk positive rate: {df['diabetes_risk'].mean():.2%}")
    print(f"Heart disease risk positive rate: {df['heart_disease_risk'].mean():.2%}")
