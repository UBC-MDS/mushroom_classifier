import pandas as pd
from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score

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