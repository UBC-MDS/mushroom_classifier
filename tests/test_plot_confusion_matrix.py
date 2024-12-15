""" Unit testing for the plot_confusion_matrix.py script """

import os
import sys
import pytest
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.plot_confusion_matrix import plot_confusion_matrix


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "season": ["s", "u", "a", "w", "u", "a"],
        "habitat": ["g", "l", "p", "u", "l", "p"],
        "cap-diameter": [1, 2, 3, 4, 5, 6],
        "class": ["p", "e", "p", "p", "e", "e"]
    })
    
@pytest.fixture
def dummy_model(sample_data):
    model = DummyClassifier(strategy="most_frequent")
    X = sample_data[["season", "habitat", "cap-diameter"]]
    y = sample_data["class"]
    model.fit(X, y)
    return model

@pytest.fixture
def temp_directory(tmp_path):
    return tmp_path


def test_plot_confusion_matrix_creates_plot(sample_data, dummy_model, temp_directory):
    """
    Test if the confusion matrix plot is created successfully.
    """
    
    X = sample_data[["season", "habitat", "cap-diameter"]]
    y = sample_data["class"]
    
    # Call the function
    plot_confusion_matrix(dummy_model, X, y, temp_directory)
    
    # Check if the file was created
    plot_path = os.path.join(temp_directory, "figures", "train_confusion_matrix.png")
    assert os.path.exists(plot_path), f"Expected plot file {plot_path} does not exist."

def test_plot_confusion_matrix_handles_empty_data(dummy_model, temp_directory):
    """
    Test if the function raises an error with empty data.
    """
    
    empty_data = pd.DataFrame(columns=["season", "habitat", "cap-diameter"])
    empty_target = pd.Series(dtype="object")
    
    with pytest.raises(ValueError):
        plot_confusion_matrix(dummy_model, empty_data, empty_target, temp_directory)

def test_plot_confusion_matrix_invalid_output_dir(dummy_model, sample_data):
    """
    Test if the function raises an error with an invalid output directory.
    """
    
    invalid_output_dir = "/invalid_directory/figures"
    
    X = sample_data[["season", "habitat", "cap-diameter"]]
    y = sample_data["class"]
    
    with pytest.raises(OSError):
        plot_confusion_matrix(dummy_model, X, y, invalid_output_dir)