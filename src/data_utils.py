import pandas as pd

def load_csv(filepath: str):
    """
    Loads a CSV file into a pandas DataFrame.
    """
    return pd.read_csv(filepath)
