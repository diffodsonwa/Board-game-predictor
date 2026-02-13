from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy import stats

def build_features(df, exclude=None):
    """
    Complete feature engineering pipeline:
    1. Rank normalization (1/rank) for columns containing 'Rank:'
    2. Skew reduction (log/sqrt based on best improvement)
    3. Min-Max scaling for numeric features (with exclusions)
    """

    # ---------------------------------------
    # 1. RANK NORMALIZATION (1 / rank value)
    # ---------------------------------------

    # Find all columns whose names contain "Rank:"
    rank_cols = [col for col in df.columns if "Rank:" in col]

    # Process each rank column
    for col in rank_cols:

        # Create a new column to store normalized 1/rank values
        df[f"{col}_norm"] = 0

        # Identify values less than 10,000 (filter unrealistic ranks)
        ranked = df[col] < 10000

        # Apply 1 / rank for valid ranks only

        df.loc[ranked, f"{col}_norm"] = 1 / df.loc[ranked, col].replace(0, np.nan)
        df[f"{col}_norm"] = df[f"{col}_norm"].fillna(0)

    # Remove the original raw rank columns
    df = df.drop(columns=rank_cols)


    # ------------------------------------------------------------
    # 2. DETERMINE BEST TRANSFORMATIONS FOR SKEW REDUCTION
    # ------------------------------------------------------------

    # Get all numeric columns (floats + ints)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Dictionary to store selected transformation for each column
    transform_dict = {}

    # Evaluate each numeric column individually
    for col in numeric_cols:

        # Replace missing values with median (required for skew calc)
        data = df[col].fillna(df[col].median())

        # Compute skewness before transformation
        skew_before = stats.skew(data)

        # Store skewness values for different methods
        transforms = {"none": skew_before}

        # Only attempt log/sqrt if values are all non-negative
        if (data >= 0).all():
            # Skew after log transform
            transforms["log"] = stats.skew(np.log1p(data))

            # Skew after square-root transform
            transforms["sqrt"] = stats.skew(np.sqrt(data))

        # Pick the method that gives the lowest absolute skew
        best = min(transforms, key=lambda x: abs(transforms[x]))

        # Only store transformation if it improves skew
        if best != "none":
            transform_dict[col] = best


    # ----------------------------------------
    # 3. APPLY CHOSEN LOG / SQRT TRANSFORMS
    # ----------------------------------------

    for col, t in transform_dict.items():

        # Apply log1p (log(1 + x)) for stability
        if t == "log":
            df[col] = np.log1p(df[col].clip(lower=0))

        # Apply square-root transform
        elif t == "sqrt":
            df[col] = np.sqrt(df[col].clip(lower=0))


    # -------------------------------
    # 4. MIN-MAX SCALING
    # -------------------------------

    # If exclude=None, turn it into an empty list
    exclude = exclude or []

    # Recompute numeric columns (some changed after transform)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Only scale numeric columns NOT in exclude list
    cols_to_scale = [c for c in numeric_cols if c not in exclude]

    # Initialize scaler
    scaler = MinMaxScaler()

    # Apply scaling column-by-column

    # Replace inf with nan, then fill nan with 0
    df[cols_to_scale] = df[cols_to_scale].replace([np.inf, -np.inf], np.nan)
    df[cols_to_scale] = df[cols_to_scale].fillna(0)
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale]) 

    # Return final engineered dataframe
    return df
