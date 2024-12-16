import click
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from data_preprocess_functions import (
    plot_target_distribution,
    create_preprocessing_pipeline,
    fit_and_save_pipeline,
    transform_and_save_data
)

def preprocess_data(input_path, output_dir, results_dir, seed):
    """
    Preprocesses the mushroom dataset, generating and saving plots, 
    preprocessor pipeline, and scaled datasets.

    Parameters:
    ----------
    input_path : str
        Path to the cleaned dataset.
    output_dir : str
        Directory to save processed datasets.
    results_dir : str
        Directory to save plots and preprocessor.
    seed : int
        Random seed for reproducibility.

    Returns:
    -------
    None
    """
    # Load cleaned data
    print(f"Loading data from {input_path}")
    mushroom_train = pd.read_csv(os.path.join(output_dir, "mushroom_train.csv"))
    mushroom_test = pd.read_csv(os.path.join(output_dir, "mushroom_test.csv"))

    y_train = mushroom_train['class']
    X_train = mushroom_train.drop(columns=['class'])
    X_test = mushroom_test.drop(columns=['class'])

    # Plot and save target variable distribution
    plot_target_distribution(y_train, results_dir)

    # Create preprocessing pipeline
    numeric_cols = ['cap-diameter', 'stem-height', 'stem-width']
    categorical_cols = [
        'cap-shape', 'cap-color', 'does-bruise-or-bleed', 'gill-color',
        'stem-color', 'has-ring', 'ring-type', 'habitat', 'season'
    ]
    impute_cols = ['ring-type']
    preprocessor = create_preprocessing_pipeline(numeric_cols, categorical_cols, impute_cols, seed)

    # Fit and save preprocessor
    fit_and_save_pipeline(preprocessor, X_train, results_dir)

    # Transform and save datasets
    transform_and_save_data(preprocessor, X_train, X_test, output_dir)

@click.command()
@click.option('--input_path', type=click.Path(exists=True), required=True, help="Path to cleaned data.")
@click.option('--output_dir', type=click.Path(), required=True, help="Directory to save processed datasets.")
@click.option('--results_dir', type=click.Path(), required=True, help="Directory to save results (plots and preprocessor).")
@click.option('--seed', type=int, default=123, help="Random seed for reproducibility.")
def main(input_path, output_dir, results_dir, seed):
    """
    Command-line interface for preprocessing data.
    """
    preprocess_data(input_path, output_dir, results_dir, seed)

if __name__ == '__main__':
    main()
