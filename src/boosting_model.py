
import pandas as pd
import numpy as np
import joblib
import inspect

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. LOAD AND PREPARE DATASET
# ==========================================

print("Loading Spotify dataset...")

df = pd.read_csv("dataset/spotify_tracks.csv")

df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df = df.dropna(
    subset=["artists", "album_name", "track_name"]
)

# ==========================================
# 2. CREATE HIT / NOT-HIT TARGET
# ==========================================

# Popularity >= 60 means Hit
df["hit"] = (df["popularity"] >= 60).astype(int)

features = [
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature"
]

X = df[features].copy()
y = df["hit"]

X["explicit"] = X["explicit"].astype(int)

print("Dataset shape:", X.shape)
print("\nTarget distribution:")
print(y.value_counts())

# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# ==========================================
# 4. CREATE BOOSTING MODEL
# ==========================================

# Use balanced class weights if supported
model_parameters = inspect.signature(
    HistGradientBoostingClassifier
).parameters

model_kwargs = {
    "random_state": 42,
    "early_stopping": True
}

if "class_weight" in model_parameters:
    model_kwargs["class_weight"] = "balanced"

boosting_model = HistGradientBoostingClassifier(
    **model_kwargs
)

# ==========================================
# 5. HYPERPARAMETER SEARCH
# ==========================================

param_distributions = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_iter": [100, 150, 200],
    "max_leaf_nodes": [15, 31, 63],
    "l2_regularization": [0.0, 0.1, 1.0],
    "min_samples_leaf": [20, 30, 50]
}

random_search = RandomizedSearchCV(
    estimator=boosting_model,
    param_distributions=param_distributions,
    n_iter=10,
    scoring="f1",
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    refit=True
)

# ==========================================
# 6. TRAIN USING 5-FOLD CROSS-VALIDATION
# ==========================================

print("\nStarting 5-fold cross-validation...")
print("Hyperparameter tuning in progress...")
print("This may take some time.")

random_search.fit(X_train, y_train)

print("\nHyperparameter tuning completed!")

# ==========================================
# 7. DISPLAY BEST PARAMETERS
# ==========================================

print("\n===== BEST PARAMETERS =====")
print(random_search.best_params_)

print("\nBest Cross-Validation F1 Score:")
print(round(random_search.best_score_, 4))

# ==========================================
# 8. EVALUATE ON TEST DATA
# ==========================================

best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== BOOSTING MODEL RESULTS =====")

print("Accuracy:", round(accuracy, 4))
print("Hit F1 Score:", round(f1, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# 9. COMPARE WITH MODULE 2 BASELINE
# ==========================================

knn_f1 = 0.3082

print("\n===== MODEL COMPARISON =====")
print("KNN Baseline F1:", knn_f1)
print("Boosting F1:", round(f1, 4))

if f1 > knn_f1:
    print("SUCCESS: Boosting beat the KNN baseline!")
else:
    print("Boosting has not beaten the KNN baseline yet.")
    print("Further tuning may be required.")

# ==========================================
# 10. SAVE BEST MODEL
# ==========================================

joblib.dump(best_model, "models/boosting_model.pkl")

print("\nBest Boosting model saved!")
print("\nBoosting training completed successfully!")
