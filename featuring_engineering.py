import os
import pandas as pd

INPUT_PATH = "/opt/ml/processing/input"
OUTPUT_PATH = "/opt/ml/processing/output"

# Load cleaned data
df = pd.read_csv(f"{INPUT_PATH}/customers_clean.csv")

# ---- Feature Engineering ----

# Drop IDs if present
if "customer_id" in df.columns:
    df = df.drop(columns=["customer_id"])

# Handle missing values
for col in df.select_dtypes(include=["number"]).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna("UNKNOWN")

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Save features
os.makedirs(OUTPUT_PATH, exist_ok=True)
df.to_csv(f"{OUTPUT_PATH}/features.csv", index=False)

print("Feature engineering completed successfully.")

