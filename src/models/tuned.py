from train import load_data, split_data, train_model_with_mlflow
from sklearn.linear_model import LassoCV

if __name__== "__main__":
    # Load and split the data
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # LassoCV aautomatically selects the best alpha

    lasso_cv  = LassoCV(alphas=None, cv=5, random_state=42, n_jobs=-1)
    lasso_cv.fit(X_train, y_train)

    best_alpha = lasso_cv.alpha_
    print(f"Best alpha found by LassoCV: {best_alpha:.4f}")

    # Create final tuned Lasso with best alpha

    from sklearn.linear_model import Lasso
    model = Lasso(alpha=best_alpha, ranodm_state=42)

    # train and log using existing train_model_with_mlflow
    train_model_with_mlflow(model)  
