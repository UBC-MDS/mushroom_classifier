import pandas as pd
from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score


def evaluate_model(model, test_data):
    """
    Evaluate the model on test data and compute evaluation metrics.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        A trained machine learning model with `score` and `predict` methods.
    test_data : pandas.DataFrame
        A DataFrame containing the test dataset. It must include a "class" column 
        with true class labels and feature columns used for prediction.

    Returns
    -------
    accuracy : float
        The accuracy of the model on the test dataset.
    f2_score : float
        The F2 score (with beta=2) of the model's predictions on the test dataset.
    evaluated_data : pandas.DataFrame
        A copy of the input `test_data` DataFrame with an additional column, "predicted",
        containing the predicted class labels.

    Examples
    --------
    >>> accuracy, f2_score, evaluated_data = evaluate_model(model, data)

    """
    print('Evaluating model on test data...')
    # Compute accuracy
    accuracy = model.score(test_data.drop(columns=["class"]), test_data["class"])

    # Compute predictions and F2 score
    test_data = test_data.assign(predicted=model.predict(test_data.drop(columns=["class"])))
    f2_score = fbeta_score(
        test_data['class'], test_data['predicted'], beta=2, pos_label='p'
    )
    return accuracy, f2_score, test_data