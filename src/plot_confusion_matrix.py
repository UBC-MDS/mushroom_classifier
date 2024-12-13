import os
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(model, data, target, output_dir):
    """Generate and save a confusion matrix plot."""
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    disp = ConfusionMatrixDisplay.from_estimator(model, data, target)
    disp.plot()
    plt.savefig(os.path.join(output_dir, "figures", "train_confusion_matrix.png"), dpi=300)
