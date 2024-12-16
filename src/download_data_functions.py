import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

def fetch_dataset(dataset_id):
    """
    Fetch dataset from UCI repository by ID.

    Parameters:
    ----------
    dataset_id : int
        The ID of the dataset to fetch.

    Returns:
    -------
    pd.DataFrame
        The fetched dataset as a DataFrame.
    """
    print(f"Fetching dataset with ID: {dataset_id}")
    dataset = fetch_ucirepo(id=dataset_id)
    raw_data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    return raw_data


def save_raw_data(data, raw_path):
    """
    Save raw dataset to a CSV file.

    Parameters:
    ----------
    data : pd.DataFrame
        The dataset to save.
    raw_path : str
        Path to save the raw dataset.

    Returns:
    -------
    None
    """
    print(f"Saving raw data to {raw_path}")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    data.to_csv(raw_path, index=False)
    print("Download complete.")
