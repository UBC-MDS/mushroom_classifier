import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from download_data_functions import fetch_dataset, save_raw_data

def test_fetch_dataset():
    """
    Test the `fetch_dataset` function by mocking the `fetch_ucirepo` API call.

    The test validates that:
    - The function correctly combines features and targets into a DataFrame.
    - The resulting DataFrame has the expected columns and data length.

    Mock Setup:
    - `fetch_ucirepo` is mocked to return a dataset with predefined features and targets.
    """
    mock_dataset = MagicMock()
    mock_dataset.data.features = pd.DataFrame({"feature1": [1, 2, 3]})
    mock_dataset.data.targets = pd.DataFrame({"target": ['a', 'b', 'c']})

    with patch('download_data_functions.fetch_ucirepo', return_value=mock_dataset):
        result = fetch_dataset(1)  # Passing a dummy dataset_id

        # Validate the structure of the result
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["feature1", "target"]
        assert len(result) == 3

def test_save_raw_data(tmp_path):
    """
    Test the `save_raw_data` function by validating the output file.

    The test checks that:
    - The file is created at the specified path.
    - The content of the saved file matches the input DataFrame.

    Mock Setup:
    - A mock DataFrame is created and saved to a temporary directory.
    """
    # Mock DataFrame
    df = pd.DataFrame({"feature1": [1, 2, 3], "target": ['a', 'b', 'c']})
    output_path = tmp_path / "raw_data.csv"

    # Call the function
    save_raw_data(df, output_path)

    # Validate file creation
    assert output_path.exists()

    # Validate file content
    saved_df = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(df, saved_df)
