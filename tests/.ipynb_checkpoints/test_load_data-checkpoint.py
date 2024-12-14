""" Unit testing for the load_data.py script """

import os
import pytest
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.load_data import load_data

@pytest.fixture
def sample_csv(tmp_path):
    """
    Fixture to create a sample CSV file for testing.
    """
    data = pd.DataFrame({
        "season": ["s", "u", "a", "w", "u", "a"],
        "habitat": ["g", "l", "p", "u", "l", "p"],
        "cap-diameter": [1, 2, 3, 4, 5, 6],
        "class": ["p", "e", "p", "p", "e"]
    })
    file_path = tmp_path / "sample_data.csv"
    data.to_csv(file_path, index=False)
    return file_path
    

def test_load_data_valid_file(sample_csv):
    """
    Test loading data from a valid CSV file.
    """
    
    data = load_data(sample_csv)
    assert isinstance(data, pd.DataFrame), "Returned data should be a pandas DataFrame."
    assert not data.empty, "The DataFrame should not be empty."
    assert list(data.columns) == ['season', 'habitat', 'cap-diameter', 'class'], "Columns mismatch."

def test_load_data_file_not_found(tmp_path):
    """
    Test loading data from a non-existing file.
    """
    
    invalid_file_path = tmp_path / "non_existent_file.csv"
    with pytest.raises(FileNotFoundError):
        load_data(invalid_file_path)

def test_load_data_empty_file(tmp_path):
    """
    Test loading data from an empty CSV file.
    """
    
    empty_file_path = tmp_path / "empty_file.csv"
    pd.DataFrame().to_csv(empty_file_path, index=False)
    data = load_data(empty_file_path)
    assert data.empty, "The DataFrame should be empty."

def test_load_data_invalid_csv(tmp_path):
    """
    Test loading data from a file that's not a valid CSV.
    """
    
    invalid_csv_path = tmp_path / "invalid.csv"
    with open(invalid_csv_path, 'w') as f:
        f.write("This is not a valid CSV format.")
    
    with pytest.raises(pd.errors.ParserError):
        load_data(invalid_csv_path)