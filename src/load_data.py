import os
import pandas as pd

def load_data(input_path):
    """
    Load the processed training data from a CSV file.

    Parameters
    ----------
    input_path : str
        The path to the processed training data CSV file.

    Returns
    -------
    pandas.DataFrame
        The loaded data as a DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file specified by `input_path` does not exist.

    Examples
    --------
    >>> df = load_data("data.csv")
    
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The file {input_path} does not exist.")
        
    try:
        data = pd.read_csv(input_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file {input_path} is empty.")
    except pd.errors.ParserError:
        raise ValueError(f"The file {input_path} is not a valid CSV.")
    if data.empty or len(data.columns) == 0:
        raise ValueError(f"The file {input_path} does not contain valid data.")
    
    return data
