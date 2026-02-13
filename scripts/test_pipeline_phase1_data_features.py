
from data.load_data import load_data
from data.data_preprocessing import preprocess_data
from feature_engineering import build_features
import os
import pandas as pd 

# Make sure Python can find your src package
import sys
sys.path.append(os.path.abspath("src")) 

# === CONFIG ===   
DATA_PATH = "/home/diffo/Documents/Data_Science_Business_Informatics/Courses/Tutorial Folder/fast_project/Game_predic_folder/src/data/DM1_game_dataset.csv" 
TARGET_COL = "Rating"  

def main():
    print("=== Testing Phase 1: Load → Preprocess → Build Features ===")

    # 1. Load Data
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"Data loaded. Shape: {df.shape}")
    print(df.head(5))
    print(df.info)
    print(df.decribe()) 

    # 2. Preprocess
    print("\n[2] Preprocessing data...")
    df_clean = preprocess_data(df, target_col=TARGET_COL)
    print(f"Data after preprocessing. Shape: {df_clean.shape}")
    print(df_clean.head(5))

    # 3. Build Features
    print("\n[3] Building features...")
    df_features = build_features(df_clean, target_col=TARGET_COL)
    print(f"Data after feature engineering. Shape: {df_features.shape}")
    print(df_features.head(3))

    print("\n Phase 1 pipeline completed successfully!")

if __name__ == "__main__":
    main() 





