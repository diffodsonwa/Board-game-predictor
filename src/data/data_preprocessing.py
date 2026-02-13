
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.decomposition import PCA

# """
# Full preprocessing pipeline:
# - Outlier capping
# - Numeric imputation
# - Categorical cleaning
# - Scaling
# - Correlation filtering
# - PCA reduction
# """

# def preprocess_data(
#     df,
#     impute_cols=None,        # List of numeric columns to impute if they contain missing values
#     exclude_scale=None,      # List of numeric columns to exclude from min-max scaling
#     corr_threshold=0.8,      # Correlation threshold for dropping highly correlated features
#     pca_variance=None        # Fraction of variance to retain in PCA, if PCA is desired
# ):
#     """
#     Apply full preprocessing pipeline on dataframe.
    
#     Returns:
#         df: Processed dataframe
#         pca_obj: PCA object if PCA applied, else None
#     """
    
#     # Make a copy to avoid modifying original
#     df = df.copy()
    
#     # ------------------- OUTLIER CAPPING -------------------
#     numeric_cols = df.select_dtypes(include="number").columns
#     for col in numeric_cols:
#         Q1, Q3 = df[col].quantile([0.25, 0.75])
#         IQR = Q3 - Q1
#         lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
#         p5, p95 = df[col].quantile([0.05, 0.95])
#         df[col] = df[col].apply(lambda x: p5 if x < lower else p95 if x > upper else x)
    
#     # ------------------- NUMERIC IMPUTATION -------------------
#     if impute_cols:
#         for col in impute_cols:
#             if col in df.columns:
#                 skew_val = df[col].skew()
#                 fill_value = df[col].median() if abs(skew_val) > 1 else df[col].mean()
#                 df[col] = df[col].fillna(fill_value)
    
#     # ------------------- CATEGORICAL CLEANING -------------------
#     if "ImagePath" in df.columns:
#         df["HasImage"] = df["ImagePath"].notna().astype(int)
#         df = df.drop(columns=["ImagePath"])
#     if "Description" in df.columns:
#         df = df[df["Description"].notna()]
#     if "Family" in df.columns:
#         df = df.drop(columns=["Family"])
    
#     # ------------------- MIN-MAX SCALING (SINGLE VERSION) -------------------
#     if exclude_scale is None:
#         exclude_scale = []
    
#     num_cols = df.select_dtypes(include=['int64', 'float64']).columns
#     cols_to_scale = [col for col in num_cols if col not in exclude_scale]
    
#     scaler = MinMaxScaler()
#     for col in cols_to_scale:
#         df[col] = scaler.fit_transform(df[[col]])
    
#     # ------------------- CORRELATION FILTER -------------------
#     # Only compute correlation on numeric columns
#     numeric_df = df.select_dtypes(include=['number'])
    
#     if numeric_df.shape[1] > 1:  # Need at least 2 columns for correlation
#         corr = numeric_df.corr()
#         high_corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
#         high_corr = high_corr.stack().reset_index()
#         high_corr.columns = ["Var1", "Var2", "Corr"]
#         high_corr = high_corr[high_corr["Corr"].abs() > corr_threshold]
#         to_drop = set(high_corr["Var2"])
#         df = df.drop(columns=[col for col in to_drop if col in df.columns])
    
#     # ------------------- PCA REDUCTION -------------------
#     pca_obj = None
#     if pca_variance:
#         # Only use numeric columns for PCA
#         X_numeric = df.select_dtypes(include=['number'])
#         if X_numeric.shape[1] > 0:
#             scaler_std = StandardScaler()
#             X_scaled = scaler_std.fit_transform(X_numeric)
#             pca = PCA(n_components=pca_variance)
#             pca.fit(X_scaled)
#             pca_obj = pca
    
#     return df, pca_obj

"""
Full preprocessing pipeline:
- Convert target column to numeric if categorical
- Outlier capping
- Numeric imputation
- Categorical cleaning
- Min-Max scaling
- Correlation filtering
- PCA reduction (optional)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

def preprocess_data(
    df,
    target_col=None,         # Name of target column (if None, no conversion)
    impute_cols=None,        # List of numeric columns to impute if they contain missing values
    exclude_scale=None,      # List of numeric columns to exclude from min-max scaling
    corr_threshold=0.8,      # Correlation threshold for dropping highly correlated features
    pca_variance=None        # Fraction of variance to retain in PCA, if PCA is desired
):
    """
    Apply full preprocessing pipeline on dataframe.
    
    Returns:
        df: Processed dataframe
        pca_obj: PCA object if PCA applied, else None
    """
    
    # Make a copy to avoid modifying original dataframe
    df = df.copy()
    
    # ------------------- CONVERT TARGET TO NUMERIC IF CATEGORICAL -------------------
    # Check if target column exists and is not already numeric
    if target_col and target_col in df.columns:
        # Check if target is categorical (object dtype or contains strings)
        if df[target_col].dtype == 'object' or pd.api.types.is_string_dtype(df[target_col]):
            # Get unique categories and sort them
            unique_vals = sorted(df[target_col].dropna().unique())
            
            # Create mapping dictionary: category -> integer (0, 1, 2, ...)
            target_map = {val: i for i, val in enumerate(unique_vals)}
            
            # Apply mapping to convert text to numbers
            df[target_col] = df[target_col].map(target_map)
            
            # Drop rows where target couldn't be mapped
            df = df.dropna(subset=[target_col])
            
            # Convert to integer then float (for regression)
            df[target_col] = df[target_col].astype(int).astype(float)
            
            print(f"✅ Converted target '{target_col}' from categorical to numeric")
            print(f"   Mapping: {target_map}")
    
    # ------------------- OUTLIER CAPPING -------------------
    # Select all numeric columns for outlier treatment
    numeric_cols = df.select_dtypes(include="number").columns
    
    # Apply outlier capping to each numeric column except target
    for col in numeric_cols:
        if target_col and col == target_col:  # Skip target column
            continue
        
        # Calculate first quartile (25th percentile)
        Q1 = df[col].quantile(0.25)
        # Calculate third quartile (75th percentile)
        Q3 = df[col].quantile(0.75)
        # Calculate Interquartile Range (IQR)
        IQR = Q3 - Q1
        # Define lower bound for outliers (Q1 - 1.5*IQR)
        lower = Q1 - 1.5 * IQR
        # Define upper bound for outliers (Q3 + 1.5*IQR)
        upper = Q3 + 1.5 * IQR
        # Calculate 5th percentile for capping low outliers
        p5 = df[col].quantile(0.05)
        # Calculate 95th percentile for capping high outliers
        p95 = df[col].quantile(0.95)
        
        # Apply lambda function to each value
        df[col] = df[col].apply(lambda x: p5 if x < lower else p95 if x > upper else x)
    
    # ------------------- NUMERIC IMPUTATION -------------------
    if impute_cols:
        for col in impute_cols:
            if col in df.columns:
                skew_val = df[col].skew()
                fill_value = df[col].median() if abs(skew_val) > 1 else df[col].mean()
                df[col] = df[col].fillna(fill_value)
    
    # ------------------- CATEGORICAL CLEANING -------------------
    if "ImagePath" in df.columns:
        df["HasImage"] = df["ImagePath"].notna().astype(int)
        df = df.drop(columns=["ImagePath"])
    if "Description" in df.columns:
        df = df[df["Description"].notna()]
    if "Family" in df.columns:
        df = df.drop(columns=["Family"])
    
    # ------------------- MIN-MAX SCALING -------------------
    if exclude_scale is None:
        exclude_scale = []
    
    # Add target to exclude_scale if it exists
    if target_col and target_col in df.columns:
        exclude_scale.append(target_col)
    
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cols_to_scale = [col for col in num_cols if col not in exclude_scale]
    
    scaler = MinMaxScaler()
    for col in cols_to_scale:
        df[col] = scaler.fit_transform(df[[col]])
    
    # ------------------- CORRELATION FILTER -------------------
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()
        upper_triangle = np.triu(np.ones(corr.shape), k=1).astype(bool)
        high_corr = corr.where(upper_triangle)
        high_corr = high_corr.stack().reset_index()
        high_corr.columns = ["Var1", "Var2", "Corr"]
        high_corr = high_corr[high_corr["Corr"].abs() > corr_threshold]
        to_drop = set(high_corr["Var2"])
        df = df.drop(columns=[col for col in to_drop if col in df.columns])
    
    # ------------------- PCA REDUCTION -------------------
    pca_obj = None
    if pca_variance:
        X_numeric = df.select_dtypes(include=['number'])
        if X_numeric.shape[1] > 0:
            scaler_std = StandardScaler()
            X_scaled = scaler_std.fit_transform(X_numeric)
            pca = PCA(n_components=pca_variance)
            pca.fit(X_scaled)
            pca_obj = pca
    
    return df, pca_obj