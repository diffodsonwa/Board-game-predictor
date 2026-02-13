#!/usr/bin/env python3
"""
run_pipeline.py
Simplest full ML workflow with automatic Lasso alpha tuning:
load → preprocess → feature engineering → tune → train → evaluate
Saves artifacts, optional plots, handles folder layouts.
"""
# The #!/usr/bin/env python3 line allows the script to be executed directly in CLI.

# -------------------------------
# Standard imports
# -------------------------------
import os              
import sys             
import pandas as pd    
import joblib          
import matplotlib.pyplot as plt  
import mlflow          
import mlflow.sklearn  
from sklearn.linear_model import LassoCV, Lasso  

# -------------------------------
# Automatically detect src folder
# -------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  
# Get the folder where this script is located (absolute path)

SRC_DIR = os.path.join(SCRIPT_DIR, "src")  
# Assume project source code is in a folder named "src" inside script folder

if not os.path.exists(SRC_DIR):
    SRC_DIR = SCRIPT_DIR  
    # If src doesn't exist, assume script itself is inside src folder

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)  
    # Add SRC_DIR to Python path so we can import modules from anywhere in src

# -------------------------------
# Import project modules
# -------------------------------
from data.load_data import load_dataset          # Function to load CSV dataset
from data.data_preprocessing import preprocess_data  # Full preprocessing pipeline
from feature_engineering import build_features   # Feature engineering pipeline
from models.train import train_model_with_mlflow, split_data  # Training utilities
from models.evaluate import evaluate_model             # Test evaluation function

# -------------------------------
# CONFIGURATION
# -------------------------------
DATA_PATH = os.path.join(SRC_DIR, "..", "data", "DM1_game_dataset.csv")  
# Path to the CSV dataset: go one level up from scripts/    



TARGET_COL = "Rating"            # Name of target column
PLOT_RESULTS = True              # Whether to plot target distribution
IMPUTE_COLS = None               # Columns to impute (None = handled automatically)
EXCLUDE_SCALE = None             # Columns to skip scaling
CORR_THRESHOLD = 0.8             # Correlation threshold for dropping highly correlated features
PCA_VARIANCE = None              # Fraction of variance to retain in PCA (None = skip)

ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")  
# Folder to save artifacts (features, PCA, model)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)  
# Create the artifacts folder if it does not exist

# -------------------------------
# Main pipeline
# -------------------------------
def main():
    print("=== RUNNING FULL PIPELINE ===")  

    # Load data
    df = load_dataset(DATA_PATH)  
    print(f"Data shape: {df.shape}")  

    #  Preprocess
    df_processed, pca_obj = preprocess_data(
        df,
        impute_cols=IMPUTE_COLS,          # Columns to impute
        exclude_scale=EXCLUDE_SCALE,      # Columns to skip scaling
        corr_threshold=CORR_THRESHOLD,    # Drop highly correlated columns above this
        pca_variance=PCA_VARIANCE         # Fraction of variance to keep (if PCA)
    )

    #  Feature engineering
    df_features = build_features(df_processed, exclude=EXCLUDE_SCALE)  
    # Apply feature transformations (rank normalization, skew reduction, scaling)

    #  Save artifacts
    feature_cols = [c for c in df_features.columns if c != TARGET_COL]  
    # Extract feature columns (exclude target)
    joblib.dump(feature_cols, os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"))  
    # Save feature columns to file
    

    if pca_obj:
        joblib.dump(pca_obj, os.path.join(ARTIFACTS_DIR, "pca_obj.pkl"))  
        # Save PCA object if PCA was applied
        

    #  Train/test split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_features)  
    # Split data into training, validation, and test sets

    #  Hyperparameter tuning (Lasso alpha)
    print("Tuning Lasso alpha with LassoCV ...")
    lasso_cv = LassoCV(alphas=None, cv=5, random_state=42, n_jobs=-1) 

    # Initialize LassoCV with 5-fold cross-validation
    lasso_cv.fit(X_train, y_train)  # Fit to training data to find best alpha
    best_alpha = lasso_cv.alpha_    # Extract best alpha value
    print(f" Best alpha found: {best_alpha:.4f}")  

    # final model
    MODEL_CLASS = Lasso                # Specify Lasso as final model
    MODEL_KWARGS = {"alpha": best_alpha}  # Use tuned alpha automatically
    print("Training model with MLflow ...")
    model = MODEL_CLASS(**MODEL_KWARGS)  
    trained_model = train_model_with_mlflow(model)  
    # Train the model and log metrics/parameters with MLflow

    # Save final model
    joblib.dump(trained_model, os.path.join(ARTIFACTS_DIR, "final_model.pkl"))  
    print(f"Final model saved")

    # Evaluate model
    print("Evaluating model on test set ...")
    evaluate_model(trained_model, X_test, y_test)  
    # Print metrics and scatter plot of predictions

    # Optional visualization
    if PLOT_RESULTS:
        import seaborn as sns
        plt.figure(figsize=(6, 4))
        sns.histplot(y_test, kde=True)  # Plot target distribution in test set
        plt.title("Target Distribution in Test Set")
        plt.show()

# -------------------------------
# Run pipeline
# -------------------------------
if __name__ == "__main__":
    main()  
