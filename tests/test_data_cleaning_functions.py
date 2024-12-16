import pandas as pd
import pytest
from data_cleaning_functions import check_missing_values, drop_columns

def test_check_missing_values(tmp_path):
    """
    Test the `check_missing_values` function by validating the output file content.

    The test checks that:
    - The output CSV file is created.
    - The content of the file correctly reflects the missing values in the input DataFrame.

    Mock Setup:
    - A DataFrame with known missing values is created and passed to the function.
    """
    # Mock DataFrame
    df = pd.DataFrame({
        "A": [1, 2, None],
        "B": [None, 2, 3],
        "C": [1, 2, 3]
    })
    output_path = tmp_path / "missing_values.csv"

    # Call the function
    check_missing_values(df, output_path)

    # Validate file creation
    assert output_path.exists()

    # Validate file content
    saved_df = pd.read_csv(output_path)
    expected_df = pd.DataFrame({
        "Column": ["A", "B", "C"],
        "Missing Count": [1, 1, 0]
    })
    pd.testing.assert_frame_equal(saved_df, expected_df)

def test_drop_columns(tmp_path):
    """
    Test the `drop_columns` function by verifying the output dataset.

    The test checks that:
    - The specified columns are removed from the dataset.
    - The output file correctly reflects the updated dataset.

    Mock Setup:
    - A DataFrame with multiple columns is created and specific columns are dropped.
    """
    # Mock DataFrame
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9]
    })
    columns_to_drop = ["B"]
    output_path = tmp_path / "cleaned.csv"

    # Call the function
    result = drop_columns(df, columns_to_drop, output_path)

    # Validate the output DataFrame
    assert "B" not in result.columns
    expected_df = df.drop(columns=columns_to_drop)
    pd.testing.assert_frame_equal(result, expected_df)

    # Validate file content
    saved_df = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(saved_df, expected_df)
