"""
HealthGuard - Health Risk Prediction Model Training
------------------------------------------------------
Trains TWO independent binary classifiers (diabetes risk &
heart disease risk) using the same patient feature set. Compares
Logistic Regression, Random Forest, and Gradient Boosting per
target, selects the best model per target by ROC-AUC, and saves
feature importance + evaluation plots.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

DATA_PATH = '/home/claude/projects/healthguard/data/patient_health_data.csv'
MODEL_DIR = '/home/claude/projects/healthguard/models'
OUTPUT_DIR = '/home/claude/projects/healthguard/outputs'

sns.set_style('whitegrid')

FEATURE_COLS = [
    'age', 'gender', 'bmi', 'systolic_bp', 'diastolic_bp', 'glucose_level',
    'cholesterol', 'heart_rate', 'physical_activity_level', 'smoker',
    'alcohol_consumption', 'family_history', 'sleep_hours', 'stress_level'
]
CATEGORICAL_COLS = ['gender', 'physical_activity_level', 'alcohol_consumption']


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


def train_for_target(df, target_col, scaler=None, fit_scaler=True):
    X = df[FEATURE_COLS]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42),
    }

    results = {}
    fitted_models = {}
    roc_data = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, roc_auc_score(y_test, y_prob))

        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob),
        }
        fitted_models[name] = model

    best_name = max(results, key=lambda k: results[k]['roc_auc'])
    best_model = fitted_models[best_name]

    return {
        'best_model_name': best_name,
        'best_model': best_model,
        'all_results': results,
        'scaler': scaler,
        'roc_data': roc_data,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test,
        'y_train': y_train,
    }


def plot_roc_curves(roc_data, target_name, save_path):
    plt.figure(figsize=(7, 6))
    colors = ['#2563eb', '#16a34a', '#dc2626']
    for (name, (fpr, tpr, auc)), color in zip(roc_data.items(), colors):
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=color, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {target_name} Prediction')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_importance(model, target_name, save_path):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return None

    fi_df = pd.DataFrame({'feature': FEATURE_COLS, 'importance': importance})
    fi_df = fi_df.sort_values('importance', ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(fi_df['feature'], fi_df['importance'], color='#0891b2')
    plt.title(f'Feature Importance - {target_name} Prediction')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return fi_df.sort_values('importance', ascending=False).to_dict(orient='records')


def plot_confusion_matrix(model, X_test, y_test, target_name, save_path):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5.5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Risk', 'At Risk'], yticklabels=['No Risk', 'At Risk'])
    plt.title(f'Confusion Matrix - {target_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    df, encoders = load_and_prepare()

    all_results = {}

    for target_col, target_name in [('diabetes_risk', 'Diabetes'), ('heart_disease_risk', 'Heart Disease')]:
        print(f"\n{'='*50}\nTraining models for: {target_name}\n{'='*50}")
        out = train_for_target(df, target_col)

        print(f"Best model: {out['best_model_name']}")
        for name, res in out['all_results'].items():
            print(f"  {name}: AUC={res['roc_auc']:.3f}, F1={res['f1_score']:.3f}, Acc={res['accuracy']:.3f}")

        plot_roc_curves(out['roc_data'], target_name, f'{OUTPUT_DIR}/roc_curve_{target_col}.png')
        fi = plot_feature_importance(out['best_model'], target_name, f'{OUTPUT_DIR}/feature_importance_{target_col}.png')
        plot_confusion_matrix(out['best_model'], out['X_test_scaled'], out['y_test'], target_name,
                               f'{OUTPUT_DIR}/confusion_matrix_{target_col}.png')

        joblib.dump(out['best_model'], f'{MODEL_DIR}/best_model_{target_col}.pkl')
        joblib.dump(out['scaler'], f'{MODEL_DIR}/scaler_{target_col}.pkl')

        all_results[target_name] = {
            'best_model': out['best_model_name'],
            'all_model_results': out['all_results'],
            'top_features': fi[:5] if fi else None,
        }

    joblib.dump(encoders, f'{MODEL_DIR}/encoders.pkl')
    joblib.dump(FEATURE_COLS, f'{MODEL_DIR}/feature_cols.pkl')

    all_results['dataset_size'] = len(df)
    all_results['diabetes_prevalence'] = float(df['diabetes_risk'].mean())
    all_results['heart_disease_prevalence'] = float(df['heart_disease_risk'].mean())

    with open(f'{OUTPUT_DIR}/model_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n\nAll models trained and saved successfully.")


if __name__ == "__main__":
    main()
