# eda.py
# Author: Yichi Zhang
# Date: 2024-12-07

import click
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def plot_feature_histograms(data, output_dir, bins=30):
    """
    Plot the histogram of each feature in the dataset by class and save the plots.

    Parameters:
        data (pd.DataFrame): The processed training data containing features and class labels.
        output_dir (str): Directory to save the plots.
        bins (int): Number of bins to use in the histogram (default: 30).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Assuming the class label column is named "class"
    if "class" not in data.columns:
        raise ValueError("The dataset does not contain a 'class' column.")
    
    features = [col for col in data.columns if col != "class" and data[col].dtype in [np.float64, np.int64]]

    for feature in features:
        plt.figure()
        for label in data["class"].unique():
            subset = data[data["class"] == label]
            subset[feature].plot(
                kind="hist",
                bins=bins,
                alpha=0.5,  # Transparency to visualize overlapping histograms
                label=f"Class {label}",
                legend=True
            )
        
        plt.title(f"Histogram for {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(axis="y")
        
        output_path = os.path.join(output_dir, f"{feature}_histogram.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot: {output_path}")


def summarize_categorical_features(data, output_dir):
    """
    Summarize categorical features in the dataset and save tables as CSV.
    
    Parameters:
        data (pd.DataFrame): The processed training data containing features and class labels.
        output_dir (str): Directory to save the summary tables.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    categorical_features = [col for col in data.columns if data[col].dtype == "object"]
    
    for feature in categorical_features:
        summary = data[feature].value_counts(normalize=False).to_frame(name="Count")
        summary["Proportion"] = data[feature].value_counts(normalize=True)
        
        output_path = os.path.join(output_dir, f"{feature}_summary.csv")
        summary.to_csv(output_path)
        print(f"Saved summary table for '{feature}' at: {output_path}")


@click.command()
@click.option('--processed-training-data', type=str, help="Path to processed training data", required=True)
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to", required=True)
@click.option('--table-to', type=str, help="Path to directory where the tables will be written to", required=True)
def main(processed_training_data, plot_to, table_to):
    """
    Main function to plot densities of numerical features and summarize categorical features in the data.
    
    Parameters:
        processed_training_data (str): Path to the processed training data CSV.
        plot_to (str): Directory where plots and tables will be saved.
    """
    print(f"Loading data from {processed_training_data}...")
    data = load_data(processed_training_data)
    print(f"Data loaded successfully with shape: {data.shape}")
    
    print(f"Generating feature density plots in {plot_to}...")
    plot_feature_histograms(data, plot_to)
    
    print(f"Summarizing categorical features and saving tables in {table_to}...")
    summarize_categorical_features(data, table_to)
    
    print("All plots and tables have been generated successfully.")


if __name__ == "__main__":
    main()