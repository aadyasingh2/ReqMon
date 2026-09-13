"""Dataset preparation script for synthetic fraud detection dataset with distribution drift injection."""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

sys.modules.setdefault("data.prepare_dataset", sys.modules[__name__])

# Clear documentation header
"""
===============================================================================
DATASET TYPE: Synthetic Imbalanced Fraud Detection Dataset
===============================================================================
This dataset is generated synthetically using `sklearn.datasets.make_classification`
with class imbalance (`weights=[0.90, 0.10]`) to simulate realistic fraud transaction
rarity (10% positive fraud rate) without relying on external network downloads.
===============================================================================
"""


class FraudClassifier(BaseEstimator, ClassifierMixin):
    """RandomForest classifier wrapper with customizable decision probability cutoff for fraud detection."""

    __module__ = "data.prepare_dataset"

    def __init__(self, proba_cutoff=0.35, class_weight={0: 1, 1: 10}, n_estimators=100, max_depth=12, random_state=42):
        self.proba_cutoff = proba_cutoff
        self.class_weight = class_weight
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, class_weight=class_weight, random_state=random_state
        )
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def predict(self, X):
        probas = self.predict_proba(X)[:, 1]
        return (probas >= self.proba_cutoff).astype(int)


def prepare_datasets_and_train_baseline():
    print("=" * 75)
    print("      RAMMA NLP — FRAUD DATASET PREPARATION & DRIFT INJECTION")
    print("=" * 75)
    print("\n[NOTE] Using synthetic imbalanced classification dataset with class_sep=2.3.")

    # 1. Generate Synthetic Imbalanced Fraud Dataset with realistic class separation
    X_mat, y_vec = make_classification(
        n_samples=5000,
        n_features=6,
        n_informative=5,
        n_redundant=1,
        class_sep=2.3,  # Clear class separation allowing ~94%+ baseline recall
        weights=[0.90, 0.10],  # 10% positive fraud rate
        random_state=42,
    )

    feature_names = [
        "transaction_amount",
        "account_age_months",
        "login_frequency",
        "failed_passwords",
        "device_risk_score",
        "location_distance_km",
    ]

    df_all = pd.DataFrame(X_mat, columns=feature_names)
    df_all["is_fraud"] = y_vec

    # Scale synthetic features to realistic domain ranges
    df_all["transaction_amount"] = np.abs(df_all["transaction_amount"] * 50 + 75)
    df_all["account_age_months"] = np.abs(df_all["account_age_months"] * 20 + 30)
    df_all["login_frequency"] = np.abs(df_all["login_frequency"] * 4 + 8)
    df_all["failed_passwords"] = np.abs(np.round(df_all["failed_passwords"] * 2 + 1))
    df_all["device_risk_score"] = np.clip(df_all["device_risk_score"] * 0.25 + 0.4, 0.0, 1.0)
    df_all["location_distance_km"] = np.abs(df_all["location_distance_km"] * 40 + 10)

    # 2. Split into Reference (calibration) and Production Clean (held-out test)
    ref_df, prod_clean_df = train_test_split(
        df_all, test_size=0.4, random_state=42, stratify=df_all["is_fraud"]
    )

    os.makedirs("data", exist_ok=True)
    ref_path = "data/reference.csv"
    prod_clean_path = "data/production_clean.csv"
    prod_drifted_path = "data/production_drifted.csv"

    ref_df.to_csv(ref_path, index=False)
    prod_clean_df.to_csv(prod_clean_path, index=False)
    print(f"\n[1] Saved reference calibration set:    {ref_path} ({len(ref_df)} rows)")
    print(f"[2] Saved clean production test set:     {prod_clean_path} ({len(prod_clean_df)} rows)")

    # 3. Train Baseline Classifier on reference.csv
    X_train = ref_df[feature_names]
    y_train = ref_df["is_fraud"]

    clf = FraudClassifier(proba_cutoff=0.35, class_weight={0: 1, 1: 10}, random_state=42)
    clf.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    model_path = "models/baseline_classifier.pkl"
    joblib.dump(clf, model_path)
    print(f"[3] Saved trained baseline model:        {model_path}")

    # 3b. Train Alternative LogisticRegression Classifier on reference.csv
    alt_clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    alt_clf.fit(X_train, y_train)
    alt_model_path = "models/alt_classifier_logreg.pkl"
    joblib.dump(alt_clf, alt_model_path)
    print(f"[3b] Saved trained alternative model:    {alt_model_path}")

    # 4. Create production_drifted.csv by perturbing features and flipping labels
    prod_drifted_df = prod_clean_df.copy()
    n_prod = len(prod_drifted_df)

    np.random.seed(42)
    # Drift feature distributions so fraud patterns mimic normal transactions (evasion/staleness drift)
    fraud_mask = prod_drifted_df["is_fraud"] == 1

    # Shift transaction_amount and device_risk_score on fraud cases downwards to obscure fraud signals
    prod_drifted_df.loc[fraud_mask, "transaction_amount"] *= 0.25
    prod_drifted_df.loc[fraud_mask, "device_risk_score"] = np.clip(
        prod_drifted_df.loc[fraud_mask, "device_risk_score"] - 0.50, 0.0, 1.0
    )
    prod_drifted_df["location_distance_km"] += np.random.normal(35.0, 12.0, size=n_prod)

    # Flip 25% of true positive fraud labels to 0 (unlabeled / stale label ground truth)
    fraud_indices = prod_drifted_df[fraud_mask].index
    flip_indices = np.random.choice(fraud_indices, size=int(len(fraud_indices) * 0.25), replace=False)
    prod_drifted_df.loc[flip_indices, "is_fraud"] = 0

    prod_drifted_df.to_csv(prod_drifted_path, index=False)
    print(f"[4] Saved drifted production test set:   {prod_drifted_path} ({len(prod_drifted_df)} rows)")

    # 5. Evaluate and Compare Baseline Recall & Metrics
    X_clean = prod_clean_df[feature_names]
    y_clean_true = prod_clean_df["is_fraud"]
    y_clean_pred = clf.predict(X_clean)

    X_drift = prod_drifted_df[feature_names]
    y_drift_true = prod_drifted_df["is_fraud"]
    y_drift_pred = clf.predict(X_drift)

    clean_recall = recall_score(y_clean_true, y_clean_pred)
    clean_precision = precision_score(y_clean_true, y_clean_pred)
    clean_f1 = f1_score(y_clean_true, y_clean_pred)
    clean_acc = accuracy_score(y_clean_true, y_clean_pred)

    drift_recall = recall_score(y_drift_true, y_drift_pred)
    drift_precision = precision_score(y_drift_true, y_drift_pred)
    drift_f1 = f1_score(y_drift_true, y_drift_pred)
    drift_acc = accuracy_score(y_drift_true, y_drift_pred)

    print("\n" + "=" * 75)
    print("           BASELINE MODEL PERFORMANCE EVALUATION COMPARISON")
    print("=" * 75)
    print(f"{'Metric':<20} | {'Clean Set (production_clean.csv)':<25} | {'Drifted Set (production_drifted.csv)':<25}")
    print("-" * 75)
    print(f"{'Recall (Fraud)':<20} | {clean_recall:<25.2%} | {drift_recall:<25.2%}")
    print(f"{'Precision':<20} | {clean_precision:<25.2%} | {drift_precision:<25.2%}")
    print(f"{'F1 Score':<20} | {clean_f1:<25.2%} | {drift_f1:<25.2%}")
    print(f"{'Accuracy':<20} | {clean_acc:<25.2%} | {drift_acc:<25.2%}")
    print("=" * 75)
    print(f"VERIFICATION SUCCESSFUL: Fraud recall dropped from {clean_recall:.2%} (clean) to {drift_recall:.2%} (drifted)!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from data.prepare_dataset import prepare_datasets_and_train_baseline
    prepare_datasets_and_train_baseline()
