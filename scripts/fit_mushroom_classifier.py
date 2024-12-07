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
@click.option('--processed-training-data', type=str, help="Path to processed training data")
@click.option('--preprocessor', type=str, help="Path to preprocessor object")
@click.option('--pipeline-to', type=str, help="Path to directory where the pipeline object will be written to")
@click.option('--results-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)
def main(processed_training_data, preprocessor, pipeline_to, results_to, seed):
    '''Fits a breast cancer classifier to the training data 
    and saves the pipeline object.'''
    np.random.seed(seed)
    set_config(transform_output="pandas")

    # read in data & preprocessor
    mushroom_train = pd.read_csv(processed_training_data)
    mushroom_preprocessor = pickle.load(open(preprocessor, "rb"))

    # create metrics
    scoring_metrics = {
    'accuracy':make_scorer(accuracy_score),
    'f2_score':make_scorer(fbeta_score, beta=2, pos_label='p',average='binary') 
    }
    cv_results = dict()

    # tune model and save results
    # knn model
    print('Training KNN Model...')
    knn = make_pipeline(mushroom_preprocessor, KNeighborsClassifier())
    knn_grid = {'kneighborsclassifier__n_neighbors':randint(5,1000)}
    cv_results['knn'] = RandomizedSearchCV(
        knn, knn_grid, n_iter=5, n_jobs=-1, cv=3,
        scoring=scoring_metrics, random_state=seed,
        refit='f2_score'
    ).fit(mushroom_train.drop(columns=["class"]),
          mushroom_train["class"])
    
    # logistic regression model
    print('Training Logistic Regression Model...')
    logreg = make_pipeline(mushroom_preprocessor,LogisticRegression(max_iter=5000,random_state=seed))
    logreg_grid = {'logisticregression__C':loguniform(1e-3,1e3)}
    cv_results['logreg'] = RandomizedSearchCV(
        logreg,logreg_grid,n_iter=30,n_jobs=-1,
        scoring=scoring_metrics,random_state=seed,
        refit='f2_score'
    ).fit(mushroom_train.drop(columns=["class"]),
          mushroom_train["class"])
    
    # svc model
    print('Training SVC Model...')
    svc = make_pipeline(mushroom_preprocessor,SVC(random_state=seed))
    svc_grid = {'svc__C':loguniform(1e-3,1e3),
            'svc__gamma':loguniform(1e-3,1e3)}
    cv_results['svc'] = RandomizedSearchCV(
        svc,svc_grid,n_iter=3,n_jobs=-1,cv=3,
        scoring=scoring_metrics,random_state=seed,
        refit='f2_score'
    ).fit(mushroom_train.drop(columns=["class"]),
          mushroom_train["class"])
    
    # compilng hyperparameters and scores of best models into one dataframe
    print('Compiling Results...')
    cols = ['params',
            'mean_fit_time',
            'mean_test_accuracy',
            'std_test_accuracy',
            'mean_test_f2_score',
            'std_test_f2_score']
    final_results = pd.concat(
        [pd.DataFrame(result.cv_results_).query('rank_test_f2_score == 1')[cols] for _,result in cv_results.items()]
    )
    final_results.index = ['KNN','Logisic Regression','SVC']
    final_results.to_csv(
        os.path.join(results_to, "tables", "cross_val_results.csv")
        )
    
    # save the best model
    best_model = cv_results['svc'].best_estimator_
    best_model.fit(
        mushroom_train.drop(columns=["class"]), 
        mushroom_train["class"]
        )

    with open(os.path.join(pipeline_to, "mushroom_best_model.pickle"), 'wb') as f:
        pickle.dump(best_model, f)

    disp = ConfusionMatrixDisplay.from_estimator(
        best_model,
        mushroom_train.drop(columns=["class"]),
        mushroom_train["class"]
        )
    disp.plot()
    plt.savefig(os.path.join(results_to, "figures", "train_confusion_matrix.png"), dpi=300)
    print("Finished model training")

if __name__ == '__main__':
    main()