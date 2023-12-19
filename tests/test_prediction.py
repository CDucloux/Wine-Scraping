from src.modules.ml_models.prediction import (
    init,
    recup_param,
    random_forest,
    boosting,
    ridge,
    mlp,
    knn,
    support_vector,
)

def test_init():
    """test de la fonction init()"""
    X_train, X_test, y_train, y_test, _ = init("type", "./tests/test_data.json")
    assert y_train.name == "type"
    assert y_test.name == "type"
    assert X_train.columns[0] == "name"
    assert X_test.columns[0] == "name"

def test_recup_param():
    """test de la fonction recup_param()"""
    param = recup_param("Random Forest", "type")
    assert isinstance(param["entrainement__max_depth"], int)
    assert isinstance(param["entrainement__n_estimators"], int)
    assert isinstance(param["imputation__strategy"], str)
    
    