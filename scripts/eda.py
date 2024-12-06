# eda.py
# author: Yichi Zhang
# date: 2024-12-07

import click
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@click.command()
@click.option('--processed-training-data', type=str, help="Path to processed training data")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
def main(processed_training_data, plot_to):
    '''Plots the densities of each feature in the processed training data
        by class and displays them as a grid of plots. Also saves the plot.'''

    mushroom_train = pd.read_csv(processed_training_data)

    numeric_columns = mushroom_train.select_dtypes(include='number')  # Select only numeric columns

    for column in numeric_columns.columns:
        plt.figure(figsize=(5,5))
        plt.hist(mushroom_train[column], bins=15, edgecolor='black', alpha=0.7)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')

        plt.savefig(os.path.join(plot_to, "figures", f"histogram_{column}.png"),
                    dpi=300)

    
    categorical_columns = mushroom_train.select_dtypes(include='object')  # Select only categorical columns

    for column in categorical_columns.columns:
        frequency = mushroom_train[column].value_counts()
        percentage = round(mushroom_train[column].value_counts(normalize=True) * 100, 2)
        freq_percent_df = pd.DataFrame({
            "Frequency": frequency,
            "Percentage": percentage
        })
        styled_df = freq_percent_df.style.format(
            precision=2
        ).background_gradient(
            subset=['Percentage'],
            cmap='YlOrRd'
        )
        fig, ax = plt.subplots(figsize=(6, 2))  # Adjust the figure size as needed
        ax.axis('off')  # Turn off the axes

        # Create a table from the DataFrame
        table = ax.table(
            cellText=freq_percent_df.values,
            colLabels=freq_percent_df.columns,
            rowLabels=freq_percent_df.index,
            loc='center',
            cellLoc='center'
        )

        # Style adjustments for readability
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(freq_percent_df.columns))))

        file_path = os.path.join(plot_to, "tables", f"{column}_frequency_table.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Saved styled table for '{column}'")


if __name__ == '__main__':
    main()