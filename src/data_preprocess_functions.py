import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


def plot_target_distribution(y_train, results_dir):
    """
    Generate and save the target variable distribution plot.

    Parameters:
    ----------
    y_train : pd.Series
        Target variable for training data.
    results_dir : str
        Directory to save the plot.

    Returns:
    -------
    None
    """
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    category_counts = y_train.value_counts()
    plt.figure(figsize=(8, 6))
    category_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Distribution of Target Variable', fontsize=14)
    plt.xlabel('Categories', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.savefig(os.path.join(results_dir, "figures", "target_variable_distribution.png"))
    plt.close()


def create_preprocessing_pipeline(numeric_cols, categorical_cols, impute_cols, seed):
    """
    Create a preprocessing pipeline with transformers for numeric and categorical data.

    Parameters:
    ----------
    numeric_cols : list of str
        Numeric columns to transform.
    categorical_cols : list of str
        Categorical columns to transform.
    impute_cols : list of str
        Columns that require imputation before encoding.
    seed : int
        Random seed for reproducibility.

    Returns:
    -------
    sklearn.compose.ColumnTransformer
        A fitted column transformer for preprocessing.
    """
    numeric_transformer = QuantileTransformer(output_distribution='normal', random_state=seed)
    categorical_transformer = OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False)
    impute_transformer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='missing'),
        categorical_transformer
    )

    preprocessor = make_column_transformer(
        (numeric_transformer, numeric_cols),
        (impute_transformer, impute_cols),
        (categorical_transformer, categorical_cols)
    )

    return preprocessor


def fit_and_save_pipeline(preprocessor, X_train, results_dir):
    """
    Fit the preprocessing pipeline on training data and save it to a file.

    Parameters:
    ----------
    preprocessor : sklearn.compose.ColumnTransformer
        Preprocessing pipeline to fit and save.
    X_train : pd.DataFrame
        Training data features.
    results_dir : str
        Directory to save the fitted pipeline.

    Returns:
    -------
    None
    """
    print("Fitting preprocessor on training data...")
    preprocessor.fit(X_train)

    print(f"Saving preprocessor to {results_dir}/models/mushroom_preprocessor.pickle")
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    with open(os.path.join(results_dir, "models", "mushroom_preprocessor.pickle"), 'wb') as f:
        pickle.dump(preprocessor, f)


def transform_and_save_data(preprocessor, X_train, X_test, output_dir):
    """
    Transform the training and test datasets using the preprocessor and save the results.

    Parameters:
    ----------
    preprocessor : sklearn.compose.ColumnTransformer
        Preprocessing pipeline to transform data.
    X_train : pd.DataFrame
        Training data features.
    X_test : pd.DataFrame
        Test data features.
    output_dir : str
        Directory to save the transformed datasets.

    Returns:
    -------
    None
    """
    print("Transforming train and test datasets...")
    scaled_train = preprocessor.transform(X_train)
    scaled_test = preprocessor.transform(X_test)

    print("Saving scaled datasets...")
    pd.DataFrame(scaled_train).to_csv(os.path.join(output_dir, "scaled_mushroom_train.csv"), index=False)
    pd.DataFrame(scaled_test).to_csv(os.path.join(output_dir, "scaled_mushroom_test.csv"), index=False)
 