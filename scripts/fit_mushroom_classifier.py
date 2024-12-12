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


def load_data_and_preprocessor(data_path, preprocessor_path):
    """Load the training data and the preprocessor."""
    data = pd.read_csv(data_path)
    preprocessor = pickle.load(open(preprocessor_path, "rb"))
    return data, preprocessor


def create_scoring_metrics():
    """Define the scoring metrics for model evaluation."""
    return {
        'accuracy': make_scorer(accuracy_score),
        'f2_score': make_scorer(fbeta_score, beta=2, pos_label='p', average='binary')
    }


def train_model(model_name, pipeline, param_grid, scoring_metrics, train_data, target, seed, n_iter=10, cv=3):
    """Train a model using RandomizedSearchCV and return the fitted results."""
    print(f'Training {model_name} Model...')
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_grid, n_iter=n_iter, n_jobs=-1,
        cv=cv, scoring=scoring_metrics, random_state=seed, refit='f2_score'
    )
    search.fit(train_data, target)
    return search


def compile_results(cv_results):
    """Compile hyperparameter tuning results into a single DataFrame."""
    cols = ['params', 'mean_fit_time', 'mean_test_accuracy', 'std_test_accuracy',
            'mean_test_f2_score', 'std_test_f2_score']
    results = pd.concat(
        [pd.DataFrame(result.cv_results_).query('rank_test_f2_score == 1')[cols]
         for _, result in cv_results.items()]
    )
    results.index = ['KNN', 'Logistic Regression', 'SVC']
    return results


def save_results(results, output_dir):
    """Save the compiled results to a CSV file."""
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    results.to_csv(os.path.join(output_dir, "tables", "cross_val_results.csv"))


def save_model(model, output_dir):
    """Save the best model to a pickle file."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "mushroom_best_model.pickle"), 'wb') as f:
        pickle.dump(model, f)


def plot_confusion_matrix(model, data, target, output_dir):
    """Generate and save a confusion matrix plot."""
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    disp = ConfusionMatrixDisplay.from_estimator(model, data, target)
    disp.plot()
    plt.savefig(os.path.join(output_dir, "figures", "train_confusion_matrix.png"), dpi=300)


@click.command()
@click.option('--processed-training-data', type=str, help="Path to processed training data")
@click.option('--preprocessor', type=str, help="Path to preprocessor object")
@click.option('--pipeline-to', type=str, help="Path to directory where the pipeline object will be written to")
@click.option('--results-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)
def main(processed_training_data, preprocessor, pipeline_to, results_to, seed):
    """Main function to train and evaluate the breast cancer classifier."""
    np.random.seed(seed)
    set_config(transform_output="pandas")

    # Load data and preprocessor
    train_data, mushroom_preprocessor = load_data_and_preprocessor(processed_training_data, preprocessor)

    # Create scoring metrics
    scoring_metrics = create_scoring_metrics()
    cv_results = {}

    # Train KNN model
    knn_pipeline = make_pipeline(mushroom_preprocessor, KNeighborsClassifier())
    knn_params = {'kneighborsclassifier__n_neighbors': randint(5, 1000)}
    cv_results['knn'] = train_model("KNN", knn_pipeline, knn_params, scoring_metrics,
                                    train_data.drop(columns=["class"]), train_data["class"], seed, n_iter=5)

    # Train Logistic Regression model
    logreg_pipeline = make_pipeline(mushroom_preprocessor, LogisticRegression(max_iter=5000, random_state=seed))
    logreg_params = {'logisticregression__C': loguniform(1e-3, 1e3)}
    cv_results['logreg'] = train_model("Logistic Regression", logreg_pipeline, logreg_params, scoring_metrics,
                                       train_data.drop(columns=["class"]), train_data["class"], seed, n_iter=30)

    # Train SVC model
    svc_pipeline = make_pipeline(mushroom_preprocessor, SVC(random_state=seed))
    svc_params = {'svc__C': loguniform(1e-3, 1e3), 'svc__gamma': loguniform(1e-3, 1e3)}
    cv_results['svc'] = train_model("SVC", svc_pipeline, svc_params, scoring_metrics,
                                    train_data.drop(columns=["class"]), train_data["class"], seed, n_iter=3)

    # Compile and save results
    results = compile_results(cv_results)
    save_results(results, results_to)

    # Save the best model
    best_model = cv_results['svc'].best_estimator_
    save_model(best_model, pipeline_to)

    # Plot confusion matrix
    plot_confusion_matrix(best_model, train_data.drop(columns=["class"]), train_data["class"], results_to)

    print("Finished model training")


if __name__ == '__main__':
    main()
