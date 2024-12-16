import os
import pandas as pd

def check_missing_values(data, output_path):
    """
    Check and log missing values in the dataset.

    Parameters:
    ----------
    data : pd.DataFrame
        The dataset to check for missing values.
    output_path : str
        Path to save missing values report.

    Returns:
    -------
    None
    """
    print("Checking missing values...")
    missing_values = data.isnull().sum().reset_index()
    missing_values.columns = ['Column', 'Missing Count']
    print("Missing values by column:\n", missing_values)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    missing_values.to_csv(output_path, index=False)

def drop_columns(data, columns_to_drop, output_path):
    """
    Drop specified columns from the dataset and save the result.

    Parameters:
    ----------
    data : pd.DataFrame
        The dataset from which columns are to be dropped.
    columns_to_drop : list of str
        List of column names to drop.
    output_path : str
        Path to save the dataset after dropping columns.

    Returns:
    -------
    pd.DataFrame
        The dataset after dropping specified columns.
    """
    print(f"Dropping columns: {columns_to_drop}")
    cleaned_data = data.drop(columns=columns_to_drop)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned_data.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

    return cleaned_data