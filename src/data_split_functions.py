import os
import pandas as pd
import pandera as pa

def validate_schema(mushroom, results_dir):
    """
    Validate the schema of the mushroom dataset and save validation results.

    Parameters:
    ----------
    mushroom : pd.DataFrame
        The mushroom dataset to validate.
    results_dir : str
        Directory to save validation results.

    Returns:
    -------
    pd.DataFrame
        Validated mushroom dataset.
    """
    print("Validating data and saving schema validation results...")
    schema = pa.DataFrameSchema(
        {
            "class": pa.Column(str, pa.Check.isin(['e', 'p'])),
            "cap-diameter": pa.Column(float, nullable=True),
            "cap-shape": pa.Column(str, nullable=True),
            "cap-color": pa.Column(str, nullable=True),
            "does-bruise-or-bleed": pa.Column(str, nullable=True),
            "gill-color": pa.Column(str, nullable=True),
            "stem-height": pa.Column(float, nullable=True),
            "stem-width": pa.Column(float, nullable=True),
            "stem-color": pa.Column(str, nullable=True),
            "has-ring": pa.Column(str, nullable=True),
            "ring-type": pa.Column(str, nullable=True),
            "habitat": pa.Column(str, nullable=True),
            "season": pa.Column(str, nullable=True)
        },
        checks=[
            pa.Check(lambda df: ~df.duplicated().any(), error="Duplicate rows found."),
            pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found.")
        ]
    )

    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
    try:
        validated_data = schema.validate(mushroom, lazy=True)
        schema_results = pd.DataFrame({"validation_status": ["Success"]})
    except pa.errors.SchemaErrors as e:
        schema_results = e.failure_cases
        print(f"Validation failed with errors:\n{schema_results}")
    schema_results.to_csv(os.path.join(results_dir, "tables", "schema_validate.csv"), index=False)

    return validated_data


def split_and_save_data(mushroom, output_dir, results_dir, seed):
    """
    Split the mushroom dataset into train and test sets, and save the results.

    Parameters:
    ----------
    mushroom : pd.DataFrame
        The mushroom dataset to split.
    output_dir : str
        Directory to save train and test datasets.
    results_dir : str
        Directory to save numeric correlation matrix.
    seed : int
        Random seed for reproducibility.

    Returns:
    -------
    tuple
        Training and test datasets (X_train, X_test, y_train, y_test).
    """
    print("Splitting features and target...")
    X = mushroom.drop(columns=['class'])
    y = mushroom['class']

    print("Splitting data into training and test sets...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    mushroom_train = X_train.copy()
    mushroom_train['class'] = y_train
    mushroom_test = X_test.copy()
    mushroom_test['class'] = y_test

    print("Saving train and test datasets...")
    os.makedirs(output_dir, exist_ok=True)
    mushroom_train.to_csv(os.path.join(output_dir, "mushroom_train.csv"), index=False)
    mushroom_test.to_csv(os.path.join(output_dir, "mushroom_test.csv"), index=False)

    print("Generating and saving numeric correlation matrix...")
    numeric_columns = X_train.select_dtypes(include='number')
    corr_matrix = numeric_columns.corr()
    corr_matrix.to_csv(os.path.join(results_dir, "tables", "numeric_correlation_matrix.csv"))

    return X_train, X_test, y_train, y_test
