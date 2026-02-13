

#!/usr/bin/env python3
"""
prepare_processed_data.py
Load raw CSV → preprocess → feature engineering → save processed CSV
"""

import os
import sys
import pandas as pd

# -------------------------------
# Make src folder importable - ABSOLUTE PATH
# -------------------------------
SRC_DIR = "/home/diffo/Documents/Data_Science_Business_Informatics/Courses/Tutorial Folder/fast_project/Game_predic_folder/src"
sys.path.insert(0, SRC_DIR) 

# -------------------------------
# Import preprocessing and features
# -------------------------------
from data.data_preprocessing import preprocess_data
from feature_engineering.feature import build_features

# -------------------------------
# Paths
# -------------------------------
RAW = os.path.join(SRC_DIR, "data", "DM1_game_dataset.csv")           
OUT = os.path.join(SRC_DIR, "data", "processed", "game_dataset_regression.csv")  

# -------------------------------
# Load raw data
# -------------------------------
df = pd.read_csv(RAW)
print(f"Loaded raw data | Shape: {df.shape}")

# -------------------------------
# Preprocess data with target column specified
# -------------------------------
df_processed, _ = preprocess_data(df, target_col="Rating")
print(f"After preprocessing | Shape: {df_processed.shape}")

# -------------------------------
# Feature engineering
# -------------------------------
df_features = build_features(df_processed)
print(f"After feature engineering | Shape: {df_features.shape}")

# -------------------------------
# Save processed CSV
# -------------------------------
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df_features.to_csv(OUT, index=False)
print(f"✅ Processed dataset saved to {OUT}")

