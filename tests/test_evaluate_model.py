import pytest
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import fbeta_score
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.evaluate_model import evaluate_model


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "season": ["s", "u", "a", "w", "u", "a"],
        "habitat": ["g", "l", "p", "u", "l", "p"],
        "cap-diameter": [1, 2, 3, 4, 5, 6],
        "class": ["p", "e", "p", "p", "e"]
    })
    
@pytest.fixture
def dummy_model(sample_data):
    model = DummyClassifier(strategy="most_frequent")
    X = sample_data[["season", "habitat", "cap-diameter"]]
    y = sample_data["class"]
    model.fit(X, y)
    return model

@pytest.fixture
def valid_test_data():
    return pd.DataFrame({
        "season": ["s", "a"],
        "habitat": ["g", "l"],
        "cap-diameter": [1, 6],
        "class": ["p", "e"]
    })



def test_evaluate_model_valid_input(dummy_model, valid_test_data):
    """
    Test function with valid inputs.
    """
    accuracy, f2, evaluated_data = evaluate_model(dummy_model, valid_test_data)
    assert isinstance(accuracy, float), "Accuracy should be a float."
    assert isinstance(f2, float), "F2 score should be a float."
    assert "predicted" in evaluated_data.columns, "Evaluated data should include 'predicted' column."

    expected_f2 = fbeta_score(valid_test_data['class'], evaluated_data['predicted'], beta=2, pos_label='p')
    assert pytest.approx(f2) == expected_f2, "F2 score does not match the expected value."


def test_evaluate_model_missing_class_column(dummy_model, valid_test_data):
    """
    Test function with missing 'class' column.
    """
    invalid_data = valid_test_data.drop(columns=["class"])
    with pytest.raises(KeyError):
        evaluate_model(dummy_model, invalid_data)


def test_evaluate_model_empty_data(dummy_model):
    """
    Test function with an empty test dataset.
    """
    empty_data = pd.DataFrame(columns=["season", "habitat", "cap-diameter", "class"])
    accuracy, f2, evaluated_data = evaluate_model(dummy_model, empty_data)
    assert accuracy == 0, "Accuracy for empty dataset should be 0."
    assert f2 == 0, "F2 score for empty dataset should be 0."
    assert evaluated_data.empty, "Evaluated data should be empty."


def test_evaluate_model_invalid_model(sample_data):
    """
    Test function with a model missing `predict` method.
    """
    class InvalidModel:
        def score(self, X, y):
            return 0.5

    invalid_model = InvalidModel()
    with pytest.raises(AttributeError):
        evaluate_model(invalid_model, sample_data)