import pandas as pd
import pytest
from data_split_functions import validate_schema, split_and_save_data
from unittest.mock import patch

def test_validate_schema(tmp_path):
    """
    Test the `validate_schema` function by ensuring validation results are correctly saved.

    The test checks that:
    - A validation result file is created.
    - The result correctly reflects schema validation success or failure.

    Mock Setup:
    - A valid DataFrame is created to pass schema validation.
    """
    # Mock DataFrame
    df = pd.DataFrame({
        "class": ["e", "p", "e"],
        "cap-diameter": [3.5, 4.0, 2.8],
        "cap-shape": ["x", "b", "x"],
        "cap-color": ["n", "g", "n"],
        "does-bruise-or-bleed": ["f", "t", "f"],
        "gill-color": ["k", "n", "k"],
        "stem-height": [5.0, 5.5, 4.7],
        "stem-width": [1.0, 1.2, 0.8],
        "stem-color": ["w", "w", "w"],
        "has-ring": ["t", "t", "t"],
        "ring-type": ["p", "e", "p"],
        "habitat": ["g", "m", "g"],
        "season": ["s", "a", "s"]
    })

    # Path to save schema validation results
    output_path = tmp_path / "schema_validate.csv"

    # Call the function
    validated_df = validate_schema(df, str(tmp_path))

    # Validate file creation
    assert output_path.exists()

    # Validate content
    saved_df = pd.read_csv(output_path)
    assert "validation_status" in saved_df.columns
    assert "Success" in saved_df["validation_status"].values

def test_split_and_save_data(tmp_path):
    """
    Test the `split_and_save_data` function by verifying file outputs.

    The test checks that:
    - Train and test datasets are created and saved.
    - Numeric correlation matrix is generated and saved.

    Mock Setup:
    - A DataFrame with valid numeric and categorical columns is created.
    """
    # Mock DataFrame
    df = pd.DataFrame({
        "class": ["e", "p", "e", "p"],
        "cap-diameter": [3.5, 4.0, 2.8, 3.2],
        "stem-height": [5.0, 5.5, 4.7, 5.2],
        "stem-width": [1.0, 1.2, 0.8, 1.1],
        "cap-color": ["n", "g", "n", "y"]
    })

    # Paths
    output_dir = tmp_path / "output"
    results_dir = tmp_path / "results"

    # Call the function
    split_and_save_data(df, str(output_dir), str(results_dir), seed=123)

    # Validate train and test datasets
    train_path = output_dir / "mushroom_train.csv"
    test_path = output_dir / "mushroom_test.csv"
    assert train_path.exists()
    assert test_path.exists()

    # Validate correlation matrix
    corr_matrix_path = results_dir / "tables" / "numeric_correlation_matrix.csv"
    assert corr_matrix_path.exists()

    # Check correlation matrix content
    corr_matrix = pd.read_csv(corr_matrix_path)
    assert "cap-diameter" in corr_matrix.columns
    assert "stem-height" in corr_matrix.columns
