
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/spotify_tracks.csv")

print("Original dataset shape:", df.shape)

# ==========================================
# 2. REMOVE UNNECESSARY COLUMN
# ==========================================

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# ==========================================
# 3. HANDLE MISSING VALUES
# ==========================================

df = df.dropna(
    subset=["artists", "album_name", "track_name"]
)

print("Shape after removing missing metadata:", df.shape)

# ==========================================
# 4. CREATE HIT / NOT-HIT TARGET
# ==========================================

HIT_THRESHOLD = 60

df["hit"] = (
    df["popularity"] >= HIT_THRESHOLD
).astype(int)

print("\nTarget distribution:")
print(df["hit"].value_counts())

print("\nTarget percentages:")
print(
    (df["hit"].value_counts(normalize=True) * 100).round(2)
)

# ==========================================
# 5. SELECT INPUT FEATURES
# ==========================================

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

# Convert boolean feature into integer
X["explicit"] = X["explicit"].astype(int)

# Target variable
y = df["hit"]

print("\nInput feature shape:", X.shape)
print("Target shape:", y.shape)

# ==========================================
# 6. TRAIN-TEST SPLIT (80/20)
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n===== TRAIN-TEST SPLIT =====")

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nData preparation completed successfully!")
