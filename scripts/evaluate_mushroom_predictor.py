'''
Evaluate the best-performing mushroom classifier on the test data, 
summarize via a confusion matrix.
'''

import click
import os
import sys
import pickle
import json
import logging
import pandas as pd
import numpy as np
import pandera as pa
from pandera import Check
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

sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from src.plot_confusion_matrix import plot_confusion_matrix
from src.evaluate_model import evaluate_model
from src.load_data import load_data


def load_pipeline(pipeline_path):
    """Load the trained pipeline object."""
    with open(pipeline_path, 'rb') as f:
        return pickle.load(f)


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



@click.command()
@click.option('--cleaned-test-data', type=str, help="Path to cleaned test data")
@click.option('--pipeline-from', type=str, help="Path to directory where the fit pipeline object lives")
@click.option('--results-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)
def main(cleaned_test_data, pipeline_from, results_to, seed):
    """Main function to evaluate the mushroom classifier."""
    np.random.seed(seed)
    set_config(transform_output="pandas")

    # Load data and pipeline
    test_data = load_data(cleaned_test_data)
    trained_pipeline = load_pipeline(pipeline_from)

    # Evaluate the model
    accuracy, f2_score, evaluated_data = evaluate_model(trained_pipeline, test_data)

    # Save results
    save_evaluation_results(accuracy, f2_score, results_to)
    save_confusion_matrix(evaluated_data, results_to)

    print("Evaluation complete.")


if __name__ == '__main__':
    main()
