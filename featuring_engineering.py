import os, glob
import pandas as pd

INPUT = "/opt/ml/processing/input"
OUTPUT = "/opt/ml/processing/output"
os.makedirs(OUTPUT, exist_ok=True)

# Load clean data
files = glob.glob(f"{INPUT}/*.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Basic feature engineering
for c in df.select_dtypes(include=["number"]).columns:
    df[c] = df[c].fillna(df[c].median())

for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].fillna("UNKNOWN")

df_features = pd.get_dummies(df, drop_first=True)

# Save features
df_features.to_csv(f"{OUTPUT}/features.csv", index=False)

print("Feature engineering complete")
print("Rows:", df_features.shape[0])
print("Columns:", df_features.shape[1])
