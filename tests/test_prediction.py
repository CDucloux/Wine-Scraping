from sklearn.pipeline import Pipeline
from src.modules.ml_models.prediction import (
    init,
    _recup_param,
    random_forest,
    boosting,
    ridge,
    mlp,
    knn,
    support_vector,
    performance
)

def test_init():
    """test de la fonction init()"""
    X_train, X_test, y_train, y_test, _ = init("type")
    assert y_train.name == "type"
    assert y_test.name == "type"
    assert X_train.columns[0] == "name"
    assert X_test.columns[0] == "name"

def test_recup_param():
    """test de la fonction recup_param()"""
    param = _recup_param("Random Forest", "type")
    assert isinstance(param["entrainement__max_depth"], int)
    assert isinstance(param["entrainement__n_estimators"], int)
    assert isinstance(param["imputation__strategy"], str)

def test_random_forest():
    """test de la fonction random_forest()"""
    model = random_forest("type", "Random Forest")
    assert isinstance(model, Pipeline)

def test_boosting():
    """test de la fonction boosting()"""
    model = boosting("type", "Boosting")
    assert isinstance(model, Pipeline)

def test_ridge():
    """test de la fonction ridge()"""
    model = ridge("type", "Ridge")
    assert isinstance(model, Pipeline)

def test_mlp():
    """test de la fonction mlp()"""
    model = mlp("type", "Réseaux de neurones")
    assert isinstance(model, Pipeline)

def test_knn():
    """test de la fonction knn()"""
    model = knn("type", "K Neighbors")
    assert isinstance(model, Pipeline)

def test_support_vector():
    """test de la fonction support_vector()"""
    model = support_vector("type", "Support Vector")
    assert isinstance(model, Pipeline)

def test_performance():
    """test de la fonction performance()"""
    score = performance("type")
    assert 0 <= score[0] <= 1
    