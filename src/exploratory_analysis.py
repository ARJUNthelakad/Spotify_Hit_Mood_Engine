
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset/spotify_tracks.csv")

# Remove unnecessary index column
df = df.drop(columns=["Unnamed: 0"])

# =================================
# 1. BASIC STATISTICS
# =================================

print("\n===== BASIC STATISTICS =====")
print(df.describe())

# =================================
# 2. CHECK MISSING VALUES
# =================================

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Display rows containing missing values
print("\n===== ROWS WITH MISSING VALUES =====")
print(df[df.isnull().any(axis=1)])

# =================================
# 3. POPULARITY DISTRIBUTION
# =================================

plt.figure(figsize=(10, 5))
sns.histplot(df["popularity"], bins=30, kde=True)

plt.title("Spotify Track Popularity Distribution")
plt.xlabel("Popularity")
plt.ylabel("Number of Songs")

plt.tight_layout()
plt.show()

# =================================
# 4. GENRE DISTRIBUTION
# =================================

print("\n===== NUMBER OF UNIQUE GENRES =====")
print(df["track_genre"].nunique())

print("\n===== TOP 10 GENRES =====")
print(df["track_genre"].value_counts().head(10))

# =================================
# 5. FEATURE CORRELATION
# =================================

numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(12, 8))
sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    annot=False
)

plt.title("Spotify Audio Features Correlation")
plt.tight_layout()
plt.show()

# =================================
# 6. POPULARITY ANALYSIS
# =================================

print("\n===== POPULARITY STATISTICS =====")

print(df["popularity"].describe())

print("\nSongs with popularity 0:")
print((df["popularity"] == 0).sum())

print("\nPopularity Percentiles:")
print(df["popularity"].quantile([0.25, 0.50, 0.75, 0.90, 0.95]))

# =================================
# 7. COMPARE HIT THRESHOLDS
# =================================

for threshold in [60, 65, 70, 75, 80]:

    hits = (df["popularity"] >= threshold).sum()
    not_hits = (df["popularity"] < threshold).sum()

    print(f"\nThreshold: {threshold}")
    print("Hit Songs:", hits)
    print("Not Hit Songs:", not_hits)
    print("Hit Percentage:", round(hits / len(df) * 100, 2), "%")
