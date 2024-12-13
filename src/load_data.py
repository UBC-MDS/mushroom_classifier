import os
import pandas as pd

def load_data(input_path):
    """
    Load the processed training data from the specified path.
    
    Parameters:
        input_path (str): Path to the processed training data CSV.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The file {input_path} does not exist.")
    return pd.read_csv(input_path)