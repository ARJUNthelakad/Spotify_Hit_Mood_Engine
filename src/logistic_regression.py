
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================

df = pd.read_csv("dataset/spotify_tracks.csv")

# Remove unnecessary column
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Remove rows with missing song metadata
df = df.dropna(
    subset=["artists", "album_name", "track_name"]
)

# Create target variable
df["hit"] = (df["popularity"] >= 60).astype(int)

# Select audio features
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
X["explicit"] = X["explicit"].astype(int)

y = df["hit"]

# ==========================================
# 2. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# 3. CREATE LOGISTIC REGRESSION PIPELINE
# ==========================================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# ==========================================
# 4. TRAIN MODEL
# ==========================================

print("Training Logistic Regression...")

model.fit(X_train, y_train)

print("Training completed!")

# ==========================================
# 5. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 6. EVALUATE MODEL
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== LOGISTIC REGRESSION RESULTS =====")

print("Accuracy:", round(accuracy, 4))
print("F1 Score:", round(f1, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# 7. SAVE MODEL
# ==========================================

joblib.dump(model, "models/logistic_model.pkl")

print("\nModel saved successfully!")
