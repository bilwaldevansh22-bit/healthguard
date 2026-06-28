# HealthGuard — Diabetes & Heart Disease Risk Predictor

HealthGuard is a dual-output machine learning system that predicts a patient's risk of diabetes and heart disease from routine clinical and lifestyle indicators, supporting early preventive screening.

## Problem Statement

Many chronic diseases like diabetes and heart disease are preventable if risk is identified early. HealthGuard uses easily-collected health metrics (BMI, blood pressure, glucose, lifestyle habits) to flag patients who may need closer medical attention — before symptoms become severe.

## Dataset

A synthetic dataset of 4,000 patient records was generated with clinically realistic feature correlations (e.g., glucose rising with BMI and age, blood pressure rising with age, compounding risk from smoking + family history + inactivity), modeled on patterns seen in datasets like Pima Indians Diabetes and UCI Heart Disease. Features include:

- Vitals: age, gender, BMI, systolic/diastolic blood pressure, glucose level, cholesterol, heart rate
- Lifestyle: physical activity level, smoking status, alcohol consumption, sleep hours, stress level
- History: family history of disease
- Targets: `diabetes_risk`, `heart_disease_risk` (independent binary labels)

## Approach

Two independent binary classification models were trained (one per condition), each comparing:
- Logistic Regression (with class balancing)
- Random Forest
- Gradient Boosting

Best model per target selected by ROC-AUC on a held-out test set.

## Results

**Diabetes Risk Prediction** — Best model: **Logistic Regression**

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.672 | 0.627 | **0.730** |
| Random Forest | 0.652 | 0.584 | 0.715 |
| Gradient Boosting | 0.659 | 0.544 | 0.710 |

**Heart Disease Risk Prediction** — Best model: **Logistic Regression**

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.671 | 0.550 | **0.762** |
| Random Forest | 0.706 | 0.536 | 0.741 |
| Gradient Boosting | 0.738 | 0.426 | 0.737 |

Top predictive features for both conditions align with established clinical risk factors (glucose level & BMI for diabetes; blood pressure, smoking, and family history for heart disease) — see `outputs/feature_importance_*.png`.

## Project Structure

```
healthguard/
├── data/
│   └── patient_health_data.csv   # Synthetic patient dataset
├── src/
│   ├── generate_data.py           # Dataset generation script
│   ├── train_model.py             # Training + evaluation for both targets
│   └── predict.py                 # Inference on a new patient
├── models/                        # Saved models, scalers, encoders (per target)
├── outputs/                       # ROC curves, confusion matrices, feature importance, metrics
└── requirements.txt
```

## How to Run

```bash
pip install -r requirements.txt
python src/generate_data.py   # generates data/patient_health_data.csv
python src/train_model.py     # trains both models, saves plots + metrics
python src/predict.py         # runs risk prediction on a sample patient
```

## Tech Stack

Python · pandas · scikit-learn · matplotlib · seaborn · joblib

## Disclaimer

This project uses synthetic data for educational/demonstration purposes only. It is **not** a medical diagnostic tool and should not be used for real clinical decision-making.

## Future Improvements

- Validate on real clinical datasets (Pima Indians Diabetes, UCI Heart Disease)
- Add SHAP explainability for individual patient predictions
- Build a simple risk-assessment web form (Streamlit) for live demos
- Explore ensemble/stacking models to push ROC-AUC higher
