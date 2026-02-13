
# Testpipeline_phase2_modeling.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import mlflow
import mlflow.sklearn
from models.train import load_data, split_data, train_model_with_mlflow  
from models.evaluate import evaluate_model 


# Load and split data

df = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

# -------------  Hyperparameter Tuning ---------------------
#  LassoCV automatically selects the best alpha using cross-validation

lasso_cv = LassoCV(alphas=None, cv = 5, random_state=42, n_jobs=-1)
lasso_cv.fit(X_train, y_train)


best_alpha = lasso_cv.alpha_
print(f"Best alpha found by LassoCV: {best_alpha:.4f}")

# -------------  Final Model ----------------------

model = Lasso(alpha=best_alpha, random_state=42)

# Train and log with MLflow 

train_model_with_mlflow(model)

# ___________________ Test evaluation ______________________

# Evaluate on test set
y_pred = model.predict(X_test)
evaluate_model(model, X_test, y_test)  
