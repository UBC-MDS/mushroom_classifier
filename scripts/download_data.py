import click
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from download_data_functions import fetch_dataset, save_raw_data

@click.command()
@click.option('--dataset_id', type=int, required=True, help="ID of the dataset to fetch from UCI repository.")
@click.option('--raw_path', type=click.Path(), default="data/raw/mushroom_raw.csv", help="Path to save the raw dataset.")
def main(dataset_id, raw_path):
    """
    Command-line interface for downloading and saving raw dataset.

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
    raw_data = fetch_dataset(dataset_id)

    # Save raw data
    save_raw_data(raw_data, raw_path)

if __name__ == '__main__':
    main()
