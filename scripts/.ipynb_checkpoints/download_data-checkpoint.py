import os
import click
import pandas as pd
from ucimlrepo import fetch_ucirepo


def fetch_and_save_ucirepo(dataset_id, raw_path):
    """
    Fetches a dataset from UCI repository and saves raw data.

    Parameters:
    ----------
    dataset_id : int
        The ID of the dataset in the UCI repository.
    raw_path : str
        Path to save the raw dataset as a CSV file.

    Returns:
    -------
    None
    """
    # Fetch dataset
    print(f"Fetching dataset with ID: {dataset_id}")
    dataset = fetch_ucirepo(id=dataset_id)
    raw_data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    # Ensure directory exists
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    # Save raw data
    print(f"Saving raw data to {raw_path}")
    raw_data.to_csv(raw_path, index=False)
    print("Download complete.")


@click.command()
@click.option('--dataset_id', type=int, required=True, help="ID of the dataset to fetch from UCI repository.")
@click.option('--raw_path', type=click.Path(), default="data/raw/mushroom_raw.csv", help="Path to save the raw dataset.")
def main(dataset_id, raw_path):
    """
    Command-line interface for downloading and saving raw dataset.
    """
    fetch_and_save_ucirepo(dataset_id, raw_path)


if __name__ == '__main__':
    main()