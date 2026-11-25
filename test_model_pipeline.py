
import numpy as np

from sklearn.metrics import accuracy_score

from model_pipeline import (
    train_model,
    evaluate_model,
    save_model,
    load_model
)


def test_train_model():

    X_train = np.array([
        [20, 1],
        [30, 0],
        [40, 1],
        [50, 0],
        [25, 1],
        [45, 0]
    ])

    y_train = np.array([0, 1, 1, 0, 0, 1])

    model = train_model(X_train, y_train)

    assert model is not None

    predictions = model.predict(X_train)

    assert len(predictions) == len(y_train)


def test_evaluate_model():

    X_train = np.array([
        [20, 1],
        [30, 0],
        [40, 1],
        [50, 0],
        [25, 1],
        [45, 0]
    ])

    y_train = np.array([0, 1, 1, 0, 0, 1])

    model = train_model(X_train, y_train)

    accuracy = evaluate_model(model, X_train, y_train)

    assert 0 <= accuracy <= 1


def test_save_and_load_model(tmp_path):

    X_train = np.array([
        [20, 1],
        [30, 0],
        [40, 1],
        [50, 0]
    ])

    y_train = np.array([0, 1, 1, 0])

    model = train_model(X_train, y_train)

    filename = str(tmp_path / "test_model.joblib")

    save_model(model, filename)

    loaded_model = load_model(filename)

    assert loaded_model is not None

    predictions = loaded_model.predict(X_train)

    assert len(predictions) == len(y_train)
