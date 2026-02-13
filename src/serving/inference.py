"""
Inference pipeline for predicting game rating.
Includes model loading and preprocessing. 
"""

import joblib
import numpy as np
import pandas as pd
import os 

# -------------------------------
# Paths to artifacts
# -------------------------------

# ART_DIR: folder where your pipeline saved artifacts
# ART_DIR = "/home/diffo/Documents/Data_Science_Business_Informatics/Courses/Tutorial Folder/fast_project/Game_predic_folder/scripts/artifacts"

ART_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "artifacts")

# MODEL_PATH: path to trained model
MODEL_PATH = os.path.join(ART_DIR, "final_model.pkl")

# FEATURES_PATH: path to saved feature columns
FEATURES_PATH = os.path.join(ART_DIR, "feature_columns.pkl")  # fixed typo: joint → join

# -------------------------------
# Load trained model and feature columns
# -------------------------------

model = joblib.load(MODEL_PATH)           # Load trained model
feature_cols = joblib.load(FEATURES_PATH) # Load feature column order

# -------------------------------
# Prepare input
# -------------------------------

def prepare_input(sample: dict) -> pd.DataFrame:
    """
    Convert a single input dict to a DataFrame with correct features.
    Handles missing features by filling zeros. 
    """
    df = pd.DataFrame([sample])                   # Convert dict to single-row DataFrame
    df = df.reindex(columns=feature_cols, fill_value=0)  # Ensure exact feature order
    return df

# -------------------------------
# Prediction function
# -------------------------------

def predict_single(sample: dict) -> float:
    """
    Take a dict of raw inputs and return the model prediction.
    
    Notes:
    - .predict() returns an array even for one row
    - [0] extracts the single value, e.g., 4.5 instead of array([4.5])
    - float(pred) ensures JSON serializable numeric output
    """
    X = prepare_input(sample)
    pred = model.predict(X)[0] 
    return float(pred)
