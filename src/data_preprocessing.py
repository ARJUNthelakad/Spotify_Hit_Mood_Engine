import pandas as pd
import numpy as np

# Load Spotify dataset
df = pd.read_csv("dataset/spotify_tracks.csv")

print("\n========== SPOTIFY HIT & MOOD ENGINE ==========")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst Five Songs:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())