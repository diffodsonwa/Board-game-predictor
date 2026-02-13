import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# ==================== PATHS ====================================
DATA_PATH = "src/data/processed/game_dataset_regression.csv"
ARTIFACT_DIR = "scripts/artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "final_model.pkl")
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "feature_columns.pkl")

TARGET = "Rating"
FEATURES = [
    "GameWeight",
    "BGGId", 
    "NumWant",
    "ComAgeRec",
    "BestPlayers"
]

TEST_SIZE = 0.2
VAL_RATIO = 0.125
RANDOM_STATE = 42

# ==================== DATA LOADING =============================
def load_data():
    """Load processed dataset."""
    df = pd.read_csv(DATA_PATH)
    return df

# ==================== SPLITTING ===============================
def split_data(df):
    """Split data into train, validation and test sets."""
    X = df[FEATURES]
    y = df[TARGET]
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_RATIO, random_state=RANDOM_STATE
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# ==================== TRAINING ================================
def train_model_with_mlflow(model):
    """Train model and log with MLflow."""
    
    # Create artifacts directory if it doesn't exist
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    with mlflow.start_run():
        # Train
        model.fit(X_train, y_train)
        
        # Validation evaluation
        val_pred = model.predict(X_val)
        r2 = r2_score(y_val, val_pred)
        mse = mean_squared_error(y_val, val_pred)
        mae = mean_absolute_error(y_val, val_pred)
        
        # Log parameters
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("features", FEATURES)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("val_ratio", VAL_RATIO)
        
        # Log metrics
        mlflow.log_metric("val_r2", r2)
        mlflow.log_metric("val_mse", mse)
        mlflow.log_metric("val_mae", mae)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        
        # Save model and features for inference
        joblib.dump(model, MODEL_PATH)
        joblib.dump(FEATURES, FEATURES_PATH)
        
        print(f"✅ Model saved to: {MODEL_PATH}")
        print(f"✅ Features saved to: {FEATURES_PATH}")
        print(f"\n{type(model).__name__} Validation R²: {r2:.3f}")
        print(f"{type(model).__name__} Validation MSE: {mse:.3f}")
        print(f"{type(model).__name__} Validation MAE: {mae:.3f}")
    
    return model

# ==================== MAIN ===================================
if __name__ == "__main__":
    model = LinearRegression()
    train_model_with_mlflow(model) 