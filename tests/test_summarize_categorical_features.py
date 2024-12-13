import pytest
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.summarize_categorical_features import summarize_categorical_features

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "season": ["s", "u", "a", "w", "u", "a"],
        "habitat": ["g", "l", "p", "u", "l", "p"],
        "cap-diameter": [1, 2, 3, 4, 5, 6]
    })

@pytest.fixture
def temp_directory(tmp_path):
    return tmp_path


def test_valid_categorical_summary(sample_data, temp_directory):
    """
    Test that summaries for categorical features are correctly generated and saved.
    """
    summarize_categorical_features(sample_data, temp_directory)
    
    # Verify the directory is created
    assert os.path.exists(temp_directory)

    for feature in ["season", "habitat"]:
        file_path = os.path.join(temp_directory, f"{feature}_summary.csv")
        assert os.path.exists(file_path)

        actual_df = pd.read_csv(file_path, index_col=0) 
        expected_df = sample_data[feature].value_counts(normalize=False).to_frame(name="Count")
        expected_df["Proportion"] = sample_data[feature].value_counts(normalize=True)
        pd.testing.assert_frame_equal(expected_df, actual_df) 


def test_output_directory_creation(sample_data, temp_directory):
    """
    Test that the function creates the output directory if it does not exist.
    """
    assert not os.path.exists(temp_directory)

    summarize_categorical_features(sample_data, temp_directory)

    # Verify the directory is created
    assert os.path.exists(temp_directory)


def test_no_categorical_features(temp_directory):
    """
    Test behavior when there are no categorical features in the dataset.
    """
    data = pd.DataFrame({"numeric1": [1, 2, 3], "numeric2": [4, 5, 6]})
    summarize_categorical_features(data, temp_directory)

    # Check if any summary files were created
    files_in_dir = os.listdir(temp_directory)
    assert len(files_in_dir) == 0  # No summary files should be created
    

def test_empty_dataset(temp_directory):
    """
    Test behavior with an empty dataset.
    """
    data = pd.DataFrame()
    summarize_categorical_features(data, temp_directory)

    # Check if any summary files were created
    files_in_dir = os.listdir(temp_directory)
    assert len(files_in_dir) == 0  # No summary files should be created
    

def test_invalid_dataframe_type(temp_directory):
    invalid_dataframe = {}  # Not a pandas DataFrame
    
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
        summarize_categorical_features(data, temp_directory)


