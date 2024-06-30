

import numpy as np
import pytest
import from sar_propensity.propensity_ops as po

@pytest.fixture
def setup_data():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    class_weights = {1: 1 / np.mean(y)}
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    class_weights = {1: 1 / np.mean(y)}
    return X, y, class_weights




def test_estimate_propensity_scores(setup_data):
    X, y, _ = setup_data
    propensity_scores = po.estimate_propensity_scores(X, y)
    assert len(propensity_scores) == len(y)

    assert (propensity_scores >= 0).all() and (propensity_scores <= 1).all()


def test_compute_class_weights(setup_data):
    X, y, _ = setup_data
    propensity_scores = po.estimate_propensity_scores(X, y)
    class_weights = po.compute_class_weights(y, propensity_scores)
    assert isinstance(class_weights, dict)
    assert 1 in class_weights

    assert isinstance(class_weights, dict)
    assert 1 in class_weights

def test_train_pu_model(setup_data):
    X, y, class_weights = setup_data
    pu_model = po.train_pu_model(X, y, class_weights)
    assert pu_model is not None

    pu_model = po.train_pu_model(X, y, class_weights)
    assert pu_model is not None


def test_calculate_optimal_threshold(setup_data):
    X, y, class_weights = setup_data
    pu_model = po.train_pu_model(X, y, class_weights)
    y_probs = pu_model.predict_proba(X)[:, 1]
    optimal_threshold = po.calculate_optimal_threshold(y, y_probs)
    assert 0 <= optimal_threshold <= 1
    pu_model = po.train_pu_model(X, y, class_weights)
    y_probs = pu_model.predict_proba(X)[:, 1]
