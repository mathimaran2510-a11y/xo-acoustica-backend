import joblib
import json
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from app.config import (
    TRAINING_CSV,
    MODEL_PATH,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TEST_SIZE,
)

# -----------------------
# Load Dataset
# -----------------------

df = pd.read_csv(TRAINING_CSV)

print(f"\nDataset Size : {len(df)}")

X = df[FEATURE_COLUMNS]
y = df["stress"]

# -----------------------
# Split
# -----------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

# -----------------------
# Improved Random Forest
# -----------------------

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# -----------------------
# Cross Validation
# -----------------------

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy",
)

print("\n5 Fold Cross Validation")

print(scores)

print(f"\nAverage CV Accuracy : {scores.mean()*100:.2f}%")

# -----------------------
# Train
# -----------------------

model.fit(X_train, y_train)

# -----------------------
# Test
# -----------------------

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("\nTest Accuracy")

print(f"{accuracy*100:.2f}%")

print("\nClassification Report")

print(classification_report(y_test, pred))

print("\nConfusion Matrix")

print(confusion_matrix(y_test, pred))

# -----------------------
# Save Model
# -----------------------

joblib.dump(model, MODEL_PATH)

print("\nModel Saved")

print(MODEL_PATH)

# -----------------------
# Feature Importance
# -----------------------

print("\nFeature Importance")

importance = model.feature_importances_

for feature, score in zip(FEATURE_COLUMNS, importance):
    print(f"{feature:20} : {score:.4f}")

# -----------------------
# Save Metadata
# -----------------------

metadata = {
    "algorithm": "Random Forest",
    "features": FEATURE_COLUMNS,
    "cross_validation_accuracy": round(scores.mean() * 100, 2),
    "test_accuracy": round(accuracy * 100, 2),
}

with open("../models/model_info.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("\nModel metadata saved.")