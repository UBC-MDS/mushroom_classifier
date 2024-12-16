"""
Validates data and splits into training and testing sets.
"""
import click
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from data_split_functions import validate_schema, split_and_save_data

def split_data(input_path, output_dir, results_dir, seed):
    """
    Splits the cleaned mushroom dataset into training and test sets, validates schema, 
    and saves train/test datasets and numeric correlation matrix.

    Parameters:
    ----------
    input_path : str
        Path to the cleaned dataset.
    output_dir : str
        Directory to save train and test datasets.
    results_dir : str
        Directory to save schema validation and numeric correlation matrix.
    seed : int
        Random seed for reproducibility.

    Returns:
    -------
    None
    """
    # Load cleaned data
    print(f"Loading data from {input_path}")
    mushroom = pd.read_csv(input_path)

    # Remove duplicate rows
    print("Removing duplicate rows...")
    mushroom = mushroom.drop_duplicates()

    # Validate schema
    validated_data = validate_schema(mushroom, results_dir)

    # Split data and save results
    split_and_save_data(validated_data, output_dir, results_dir, seed)

@click.command()
@click.option('--input_path', type=click.Path(exists=True), required=True, help="Path to cleaned data.")
@click.option('--output_dir', type=click.Path(), required=True, help="Directory to save processed datasets.")
@click.option('--results_dir', type=click.Path(), required=True, help="Directory to save results (tables).")
@click.option('--seed', type=int, default=123, help="Random seed for reproducibility.")
def main(input_path, output_dir, results_dir, seed):
    """
    Command-line interface for splitting and validating data.
    """
    split_data(input_path, output_dir, results_dir, seed)

if __name__ == '__main__':
    main()
