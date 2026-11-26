
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/spotify_tracks.csv")

df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df = df.dropna(
    subset=["artists", "album_name", "track_name"]
)

# ==========================================
# 2. SELECT AUDIO FEATURES
# ==========================================

mood_features = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "tempo",
    "loudness"
]

X = df[mood_features].copy()

print("Mood dataset shape:", X.shape)

# ==========================================
# 3. STANDARDIZE FEATURES
# ==========================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature scaling completed!")

# ==========================================
# 4. APPLY PCA (2 COMPONENTS)
# ==========================================

pca = PCA(n_components=2, random_state=42)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA completed!")

print(
    "Explained variance ratio:",
    pca.explained_variance_ratio_
)

print(
    "Total variance explained:",
    round(sum(pca.explained_variance_ratio_) * 100, 2),
    "%"
)

# ==========================================
# 5. APPLY K-MEANS CLUSTERING
# ==========================================

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

df["mood_cluster"] = clusters

print("\nK-Means clustering completed!")

# ==========================================
# 6. DISPLAY CLUSTER DISTRIBUTION
# ==========================================

print("\n===== CLUSTER COUNTS =====")

print(
    df["mood_cluster"]
    .value_counts()
    .sort_index()
    .to_string()
)

# ==========================================
# 7. UNDERSTAND CLUSTER PROFILES
# ==========================================

print("\n===== CLUSTER FEATURE AVERAGES =====")

cluster_profiles = df.groupby(
    "mood_cluster"
)[mood_features].mean()

# Display all nine features without truncation
print(cluster_profiles.round(3).to_string())

# ==========================================
# 8. CREATE PCA VISUALIZATION
# ==========================================

plt.figure(figsize=(10, 7))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="viridis",
    s=8,
    alpha=0.5
)

plt.title("Spotify Songs - PCA 2D Mood Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Mood Cluster")

plt.tight_layout()
plt.savefig("outputs/mood_clusters.png", dpi=150)
plt.show()

# ==========================================
# 9. SAVE MODELS
# ==========================================

joblib.dump(scaler, "models/mood_scaler.pkl")
joblib.dump(kmeans, "models/kmeans_model.pkl")
joblib.dump(pca, "models/mood_pca.pkl")

print("\nMood scaler saved!")
print("K-Means model saved!")
print("PCA model saved!")

print("\nMood clustering completed successfully!")
