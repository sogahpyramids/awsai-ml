import pandas as pd
import os

INPUT_PATH = "/opt/ml/processing/input"
OUTPUT_PATH = "/opt/ml/processing/output"

# Read all parquet files in folder
df = pd.read_parquet(INPUT_PATH)

# Feature engineering
df = pd.get_dummies(df)

os.makedirs(OUTPUT_PATH, exist_ok=True)
df.to_csv(f"{OUTPUT_PATH}/features.csv", index=False)
