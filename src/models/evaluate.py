
from train import load_data, split_data
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Metrics
    print(f"Test R²: {r2_score(y_test, y_pred):.3f}")
    print(f"Test MSE: {mean_squared_error(y_test, y_pred):.3f}")
    print(f"Test MAE: {mean_absolute_error(y_test, y_pred):.3f}")

    #   Plot predictions vs actual
    plt.figure(figsize=(6,4))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.title('Actual vs Predicted Ratings')
    plt.grid(True, alpha=0.3)
    plt.show()

    
