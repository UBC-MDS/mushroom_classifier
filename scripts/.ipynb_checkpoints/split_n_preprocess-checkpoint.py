import click
import os
import pandas as pd
import pandera as pa
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


@click.command()
@click.option('--input_path', type=click.Path(exists=True), required=True, help="Path to cleaned data.")
@click.option('--output_dir', type=click.Path(), required=True, help="Directory to save processed datasets.")
@click.option('--results_dir', type=click.Path(), required=True, help="Directory to save results (plots, tables, and preprocessor).")
@click.option('--seed', type=int, default=123, help="Random seed for reproducibility.")
def main(input_path, output_dir, results_dir, seed):
    """
    Splits, validates, and preprocesses the mushroom dataset, saving processed datasets, 
    target distribution plot, schema validation results, numeric correlation matrix, and the preprocessor object.
    """
    # Load cleaned data
    print(f"Loading data from {input_path}")
    mushroom = pd.read_csv(input_path)

    # Remove duplicate rows
    print("Removing duplicate rows...")
    mushroom = mushroom.drop_duplicates()

    # Define schema for validation
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

    # Validate data and save results
    print("Validating data and saving schema validation results...")
    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
    try:
        validated_data = schema.validate(mushroom, lazy=True)
        schema_results = pd.DataFrame({"validation_status": ["Success"]})
    except pa.errors.SchemaErrors as e:
        schema_results = e.failure_cases
        print(f"Validation failed with errors:\n{schema_results}")
    schema_results.to_csv(os.path.join(results_dir, "tables", "schema_validate.csv"), index=False)

    # Split features and target
    print("Splitting features and target...")
    X = mushroom.drop(columns=['class'])
    y = mushroom['class']

    # Split data into training and test sets
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Combine features and target for saving
    mushroom_train = X_train.copy()
    mushroom_train['class'] = y_train
    mushroom_test = X_test.copy()
    mushroom_test['class'] = y_test

    # Save train and test datasets
    print("Saving train and test datasets...")
    os.makedirs(output_dir, exist_ok=True)
    mushroom_train.to_csv(os.path.join(output_dir, "mushroom_train.csv"), index=False)
    mushroom_test.to_csv(os.path.join(output_dir, "mushroom_test.csv"), index=False)

    # Generate and save numeric correlation matrix
    print("Generating and saving numeric correlation matrix...")
    numeric_columns = mushroom_train.select_dtypes(include='number')
    corr_matrix = numeric_columns.corr()
    corr_matrix.to_csv(os.path.join(results_dir, "tables", "numeric_correlation_matrix.csv"))

    # Plot and save target variable distribution
    print("Generating and saving target variable distribution plot...")
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    category_counts = y_train.value_counts()
    plt.figure(figsize=(8, 6))
    category_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Distribution of Target Variable', fontsize=14)
    plt.xlabel('Categories', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.savefig(os.path.join(results_dir, "figures", "target_variable_distribution.png"))
    plt.close()

    # Preprocessing pipeline
    print("Creating preprocessing pipeline...")
    numeric_cols = ['cap-diameter', 'stem-height', 'stem-width']
    categorical_cols = [
        'cap-shape', 'cap-color', 'does-bruise-or-bleed', 'gill-color',
        'stem-color', 'has-ring', 'ring-type', 'habitat', 'season'
    ]
    impute_cols = ['ring-type']

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

    # Fit preprocessor on training data
    print("Fitting preprocessor on training data...")
    preprocessor.fit(X_train)

    # Save the preprocessor
    print(f"Saving preprocessor to {results_dir}/models/mushroom_preprocessor.pickle")
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    with open(os.path.join(results_dir, "models", "mushroom_preprocessor.pickle"), 'wb') as f:
        pickle.dump(preprocessor, f)

    # Transform datasets
    print("Transforming train and test datasets...")
    scaled_train = preprocessor.transform(X_train)
    scaled_test = preprocessor.transform(X_test)

    # Save scaled datasets
    print("Saving scaled datasets...")
    pd.DataFrame(scaled_train).to_csv(os.path.join(output_dir, "scaled_mushroom_train.csv"), index=False)
    pd.DataFrame(scaled_test).to_csv(os.path.join(output_dir, "scaled_mushroom_test.csv"), index=False)

    print("Data splitting and preprocessing complete.")


if __name__ == '__main__':
    main()