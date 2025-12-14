# ======================================================================
# ✈ British Airways – Optimized Machine Learning Pipeline
# Booking Completion Prediction + Customer Characteristics Analysis
# ======================================================================

# ----------------------------
# Imports
# ----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from imblearn.over_sampling import SMOTE
import shap

# ======================================================================
# 1️⃣ Load Data
# ======================================================================

df = pd.read_csv(
    "C:/Users/asus/OneDrive/Desktop/British/customer_booking.csv",
    encoding="latin1"
)

# Separate features and target
X = df.drop("booking_complete", axis=1)
y = df["booking_complete"]

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# ======================================================================
# 2️⃣ Train / Test Split (Stratified)
# ======================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

# ======================================================================
# 3️⃣ Handle Class Imbalance Using SMOTE
# ======================================================================

sm = SMOTE(random_state=42)

X_train_res, y_train_res = sm.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(
    "Train Resampled:",
    len(X_train_res),
    "| Class balance:",
    y_train_res.value_counts().to_dict()
)

# ======================================================================
# 4️⃣ Hyperparameter Tuning – Random Forest
# ======================================================================

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

param_dist = {
    "n_estimators": [600, 800, 1000, 1200],
    "max_depth": [15, 20, 25, 30, None],
    "min_samples_split": [2, 4, 6, 10],
    "min_samples_leaf": [2, 3, 4, 5],
    "max_features": ["sqrt", "log2", 0.5],
    "class_weight": [
        {0: 1, 1: w} for w in [1, 2, 3, 4, 5]
    ]
}

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=25,
    scoring="f1",
    cv=4,
    n_jobs=-1,
    random_state=42
)

print("\n🔍 Running hyperparameter search...")
search.fit(X_train_res, y_train_res)

best_rf = search.best_estimator_

print("\nBest Parameters Found:")
print(search.best_params_)

# ======================================================================
# 5️⃣ Probability Calibration
# ======================================================================

calibrated_rf = CalibratedClassifierCV(
    estimator=best_rf,
    method="isotonic",
    cv=4
)

calibrated_rf.fit(
    X_train_res,
    y_train_res
)

# ======================================================================
# 6️⃣ Predict Probabilities + Optimal F1 Threshold
# ======================================================================

probs = calibrated_rf.predict_proba(X_test)[:, 1]

prec, rec, thresholds = precision_recall_curve(
    y_test,
    probs
)

f1_scores = (2 * prec * rec) / (prec + rec + 1e-9)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"\nOptimal Threshold = {best_threshold:.3f}")
print(f"Max F1 Score      = {f1_scores[best_idx]:.3f}")

y_pred = (probs >= best_threshold).astype(int)

# ======================================================================
# 7️⃣ Evaluation
# ======================================================================

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred, digits=3))

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, y_pred))

# ======================================================================
# 8️⃣ Feature Importance (from tuned RF)
# ======================================================================

importances = best_rf.feature_importances_

fi = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": importances
    }
).sort_values(
    by="importance",
    ascending=False
)

print("\nTop 15 Predictive Features:")
print(fi.head(15))

# ======================================================================
# 9️⃣ SHAP – Customer Behavior Interpretation
# ======================================================================

explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test)

# SHAP summary plot
shap.summary_plot(
    shap_values[1],
    X_test,
    show=False
)

plt.title("SHAP Summary – Key Drivers of Ticket Purchase")
plt.tight_layout()
plt.show()

# Feature importance bar plot
plt.figure(figsize=(8, 5))

plt.barh(
    fi.head(15).feature,
    fi.head(15).importance
)

plt.gca().invert_yaxis()
plt.title("Top 15 Features – Random Forest Importance")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()

print("\n✅ Complete: Optimized BA Prediction Pipeline Ready.")
