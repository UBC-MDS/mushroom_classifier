# fit_mushroom_classifier.py
# author: Yichi Zhang
# date: 2024-12-07

import click
import os
import pickle
import json
import logging
from ucimlrepo import fetch_ucirepo 
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

@click.command()
@click.option('--scaled-test-data', type=str, help="Path to scaled test data")
@click.option('--pipeline-from', type=str, help="Path to directory where the fit pipeline object lives")
@click.option('--results-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)
def main(scaled_test_data, pipeline_from, results_to, seed):
    '''Evaluates the breast cancer classifier on the test data 
    and saves the evaluation results.'''
    np.random.seed(seed)
    set_config(transform_output="pandas")

    mushroom_test = pd.read_csv(scaled_test_data)

    with open(pipeline_from, 'rb') as f:
        mushroom_fit = pickle.load(f)

    # Compute accuracy
    accuracy = mushroom_fit.score(
        mushroom_test.drop(columns=["target"]),
        mushroom_test["target"]
    )

    # Compute F2 score (beta = 2)
    mushroom_preds = mushroom_test.assign(
        predicted=mushroom_fit.predict(mushroom_test)
    )
    f2_beta_2_score = fbeta_score(
        mushroom_preds['target'],
        mushroom_preds['predicted'],
        beta=2,
        pos_label='p'
    )

    test_scores = pd.DataFrame({'accuracy': [accuracy], 
                                'F2 score (beta = 2)': [f2_beta_2_score]})
    test_scores.to_csv(os.path.join(results_to, "test_scores.csv"), index=False)

    confusion_matrix = pd.crosstab(
        mushroom_preds["target"],
        mushroom_preds["predicted"]
    )
    confusion_matrix.to_csv(os.path.join(results_to, "tables", "confusion_matrix.csv"))

    disp = ConfusionMatrixDisplay.from_predictions(
        mushroom_preds["target"],
        mushroom_preds["predicted"]
    )
    disp.plot()
    plt.savefig(os.path.join(results_to, "figures", "confusion_matrix.png"), dpi=300)


if __name__ == '__main__':
    main()
