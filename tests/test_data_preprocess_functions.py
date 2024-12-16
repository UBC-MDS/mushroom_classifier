import pandas as pd
import pytest
import numpy as np
import os
import pickle
from sklearn.compose import ColumnTransformer
from data_preprocess_functions import (
    plot_target_distribution,
    create_preprocessing_pipeline,
    fit_and_save_pipeline,
    transform_and_save_data
)

def test_plot_target_distribution(tmp_path):
    """
    Test the `plot_target_distribution` function by ensuring the plot file is created.

    The test checks that:
    - The plot file is created at the specified location.

    Mock Setup:
    - A mock target variable distribution is passed to the function.
    """
    y_train = pd.Series(["e", "e", "p", "p", "e"])
    results_dir = tmp_path / "results"

    # Call the function
    plot_target_distribution(y_train, str(results_dir))

    # Validate the file exists
    plot_path = results_dir / "figures" / "target_variable_distribution.png"
    assert plot_path.exists()

def test_create_preprocessing_pipeline():
    """
    Test the `create_preprocessing_pipeline` function by ensuring the pipeline is correctly structured.

    The test checks that:
    - The returned pipeline is an instance of `ColumnTransformer`.
    - The pipeline contains the expected transformers.
    """
    numeric_cols = ["col1", "col2"]
    categorical_cols = ["col3", "col4"]
    impute_cols = ["col5"]

    pipeline = create_preprocessing_pipeline(numeric_cols, categorical_cols, impute_cols, seed=42)

    # Validate pipeline structure
    assert isinstance(pipeline, ColumnTransformer)
    transformers = [name for name, _ in pipeline.transformers]
    assert "pipeline-1" in transformers
    assert "pipeline-2" in transformers

def test_fit_and_save_pipeline(tmp_path):
    """
    Test the `fit_and_save_pipeline` function by ensuring the pipeline is fitted and saved correctly.

    The test checks that:
    - The pipeline is saved as a `.pickle` file.

    Mock Setup:
    - A mock pipeline and dataset are used.
    """
    preprocessor = create_preprocessing_pipeline(
        numeric_cols=["cap"], categorical_cols=["stem", "gill"], impute_cols=[], seed=0
    )
    tmp_path / "models"
