import os
import click
import pandas as pd


def clean_transform_data(input_path, output_path, columns_to_drop_path, missing_values_path):
    """
    Cleans the dataset by dropping specified columns, logs missing values, 
    and saves the list of dropped columns and missing values by column.

    Parameters:
    ----------
    input_path : str
        Path to the input raw dataset.
    output_path : str
        Path to save the cleaned dataset as a CSV file.
    columns_to_drop_path : str
        Path to save the list of dropped columns as a CSV file.
    missing_values_path : str
        Path to save the missing values by column as a CSV file.

    Returns:
    -------
    None
    """
    # Load raw dataset
    print(f"Loading raw data from {input_path}")
    raw_data = pd.read_csv(input_path)

    # Check missing values and log
    print("Checking missing values...")
    missing_values = raw_data.isnull().sum().reset_index()
    missing_values.columns = ['Column', 'Missing Count']
    print("Missing values by column:\n", missing_values)

    # Save missing values by column to a CSV file
    print(f"Saving missing values to {missing_values_path}")
    os.makedirs(os.path.dirname(missing_values_path), exist_ok=True)
    missing_values.to_csv(missing_values_path, index=False)

    # Predefined list of columns to drop
    columns_to_drop = [
        'cap-surface', 'gill-attachment', 'gill-spacing',
        'stem-root', 'stem-surface', 'veil-type', 'veil-color',
        'spore-print-color'
    ]

    # Save dropped columns to a CSV file
    print(f"Saving columns to drop to {columns_to_drop_path}")
    os.makedirs(os.path.dirname(columns_to_drop_path), exist_ok=True)
    pd.DataFrame({"columns_to_drop": columns_to_drop}).to_csv(columns_to_drop_path, index=False)

    # Drop the specified columns
    print(f"Dropping columns: {columns_to_drop}")
    cleaned_data = raw_data.drop(columns=columns_to_drop)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the cleaned dataset
    print(f"Saving cleaned data to {output_path}")
    cleaned_data.to_csv(output_path, index=False)
    print("Data cleaning complete.")


@click.command()
@click.option('--input_path', type=click.Path(exists=True), required=True, help="Path to the input raw dataset.")
@click.option('--output_path', type=click.Path(), required=True, help="Path to save the cleaned dataset.")
@click.option('--columns_to_drop_path', type=click.Path(), required=True, help="Path to save the dropped columns.")
@click.option('--missing_values_path', type=click.Path(), required=True, help="Path to save the missing values by column.")
def main(input_path, output_path, columns_to_drop_path, missing_values_path):
    """
    Command-line interface for cleaning and transforming data.
    """
    clean_transform_data(input_path, output_path, columns_to_drop_path, missing_values_path)


if __name__ == '__main__':
    main()