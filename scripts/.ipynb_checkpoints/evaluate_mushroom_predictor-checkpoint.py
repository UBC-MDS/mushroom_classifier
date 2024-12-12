# fit_mushroom_classifier.py
# author: Yichi Zhang
# date: 2024-12-07

import click
import os
import pickle
import json
import logging
import pandas as pd
import numpy as np
import pandera as pa
from pandera import Check
from deepchecks import Dataset
import matplotlib.pyplot as plt
from scipy.stats import loguniform, randint
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer,OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import ConfusionMatrixDisplay, make_scorer, fbeta_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_validate, cross_val_predict, GridSearchCV, RandomizedSearchCV


def load_test_data(data_path):
    """Load the cleaned test data."""
    return pd.read_csv(data_path)


def load_pipeline(pipeline_path):
    """Load the trained pipeline object."""
    with open(pipeline_path, 'rb') as f:
        return pickle.load(f)


def evaluate_model(model, test_data):
    """Evaluate the model on test data and compute metrics."""
    print('Evaluating model on test data...')
    # Compute accuracy
    accuracy = model.score(test_data.drop(columns=["class"]), test_data["class"])

    # Compute predictions and F2 score
    test_data = test_data.assign(predicted=model.predict(test_data.drop(columns=["class"])))
    f2_score = fbeta_score(
        test_data['class'], test_data['predicted'], beta=2, pos_label='p'
    )
    return accuracy, f2_score, test_data


def save_evaluation_results(accuracy, f2_score, output_dir):
    """Save evaluation metrics to a CSV file."""
    results = pd.DataFrame({'accuracy': [accuracy], 'F2 score (beta = 2)': [f2_score]})
    os.makedirs(output_dir, exist_ok=True)
    results.to_csv(os.path.join(output_dir, "tables", "test_scores.csv"), index=False)


def save_confusion_matrix(test_data, output_dir):
    """Save the confusion matrix as a CSV file."""
    confusion_matrix = pd.crosstab(
        test_data["class"], test_data["predicted"], rownames=["Actual"], colnames=["Predicted"]
    )
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    confusion_matrix.to_csv(os.path.join(output_dir, "tables", "test_confusion_matrix.csv"))


def plot_confusion_matrix(test_data, output_dir):
    """Generate and save a confusion matrix plot."""
    disp = ConfusionMatrixDisplay.from_predictions(
        test_data["class"], test_data["predicted"]
    )
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    disp.plot()
    plt.savefig(os.path.join(output_dir, "figures", "test_confusion_matrix.png"), dpi=300)


@click.command()
@click.option('--cleaned-test-data', type=str, help="Path to cleaned test data")
@click.option('--pipeline-from', type=str, help="Path to directory where the fit pipeline object lives")
@click.option('--results-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)
def main(cleaned_test_data, pipeline_from, results_to, seed):
    """Main function to evaluate the breast cancer classifier."""
    np.random.seed(seed)
    set_config(transform_output="pandas")

    # Load data and pipeline
    test_data = load_test_data(cleaned_test_data)
    trained_pipeline = load_pipeline(pipeline_from)

    # Evaluate the model
    accuracy, f2_score, evaluated_data = evaluate_model(trained_pipeline, test_data)

    # Save results
    save_evaluation_results(accuracy, f2_score, results_to)
    save_confusion_matrix(evaluated_data, results_to)
    plot_confusion_matrix(evaluated_data, results_to)

    print("Evaluation complete.")


if __name__ == '__main__':
    main()
