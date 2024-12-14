import os
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(model, data, target, output_dir):
    """
    Generate and save a confusion matrix plot for the given model and data.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        A trained machine learning model with a `predict` method.
    data : pandas.DataFrame or numpy.ndarray
        The feature dataset used for generating predictions.
    target : pandas.Series or numpy.ndarray
        The true class labels corresponding to the `data`.
    output_dir : str
        The directory where the confusion matrix plot will be saved.

    Examples
    --------
    >>> plot_confusion_matrix(model, data, target, output_dir="results")
    
    """
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    disp = ConfusionMatrixDisplay.from_estimator(model, data, target)
    disp.plot()
    plt.savefig(os.path.join(output_dir, "figures", "train_confusion_matrix.png"), dpi=300)
