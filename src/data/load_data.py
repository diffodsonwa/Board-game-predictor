
import pandas as pd

def load_dataset(path):
    """Load CSV dataset and return dataframe."""
    df = pd.read_csv(path)
    return df



