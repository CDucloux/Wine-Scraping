from sklearn.model_selection import GridSearchCV
from src.modules.ml_models.models import (
    data_model,
    prep_str,
    model_ridge,
    model_boost,
    model_knn,
    model_mlp,
    model_rf,
    model_svm,
    train_model,
    score_entrainement,
    score_test,
    ecart_type_test,
    ecart_type_train,
    parametre,
)
from src.modules.ml_models.prediction import init


def test_data_model():
    """test la fonction data_model()"""
    data_type = data_model("./data/vins.json", "type")
    data_unit_price = data_model("./data/vins.json", "unit_price")
    assert data_type["type"].is_null().sum() == 0
    assert data_unit_price["unit_price"].is_null().sum() == 0


def test_prep_str():
    """test la fonction prep_str()"""
    data_type = data_model("./data/vins.json", "type")
    df = prep_str(data_type, ["cepage", "par_gouts", "service", "country"])
    assert data_type.shape[1] < df.shape[1]


def test_model_rf():
    """test la fonction model_rf()"""
    X_train_n, _, y_train, _, _ = init("type")
    X_train = X_train_n.drop(columns=["name"])
    model = model_rf(X_train[0:50], y_train[0:50], "classification")
    assert isinstance(model, GridSearchCV)


def test_model_boost():
    """test la fonction model_boost()"""
    X_train_n, _, y_train, _, _ = init("type")
    X_train = X_train_n.drop(columns=["name"])
    model = model_boost(X_train[0:50], y_train[0:50], "classification")
    assert isinstance(model, GridSearchCV)


def test_model_knn():
    """test la fonction model_knn()"""
    X_train_n, _, y_train, _, _ = init("type")
    X_train = X_train_n.drop(columns=["name"])
    model = model_knn(X_train[0:50], y_train[0:50], "classification")
    assert isinstance(model, GridSearchCV)


def test_model_svm():
    """test la fonction model_svm()"""
    X_train_n, _, y_train, _, _ = init("type")
    X_train = X_train_n.drop(columns=["name"])
    model = model_svm(X_train[0:50], y_train[0:50], "classification")
    assert isinstance(model, GridSearchCV)


def test_model_mlp():
    """test la fonction model_mlp()"""
    X_train_n, _, y_train, _, _ = init("type")
    X_train = X_train_n.drop(columns=["name"])
    model = model_mlp(X_train[0:50], y_train[0:50], "classification")
    assert isinstance(model, GridSearchCV)


def test_model_ridge():
    """test la fonction model_ridge()"""
    X_train_n, _, y_train, _, _ = init("type")
    X_train = X_train_n.drop(columns=["name"])
    model = model_ridge(X_train[0:50], y_train[0:50], "classification")
    assert isinstance(model, GridSearchCV)


def test_train_model():
    """test la fonction train_model()"""
    X_train_n, _, y_train, _, _ = init("type")
    X_train = X_train_n.drop(columns=["name"])
    model = train_model(X_train[0:50], y_train[0:50], "classification")
    assert isinstance(model, dict)


def test_score():
    """test les fonctions score_test() et score_entrainement()"""
    X_train_n, _, y_train, _, _ = init("type")
    X_train = X_train_n.drop(columns=["name"])
    model = model_knn(X_train[0:50], y_train[0:50], "classification")
    score_t = score_test(model)
    score_e = score_entrainement(model)
    assert 0 <= score_t <= 1
    assert 0 <= score_e <= 1


def test_ecart_type():
    """test les fonctions ecart_type_test() et ecart_type_train()"""
    X_train_n, _, y_train, _, _ = init("type")
    X_train = X_train_n.drop(columns=["name"])
    model = model_knn(X_train[0:50], y_train[0:50], "classification")
    ecart_t = ecart_type_test(model)
    ecart_e = ecart_type_train(model)
    assert isinstance(ecart_t, float)
    assert isinstance(ecart_e, float)


def test_parametre():
    """test la fonctions test_parametre()"""
    X_train_n, _, y_train, _, _ = init("type")
    X_train = X_train_n.drop(columns=["name"])
    model = model_knn(X_train[0:50], y_train[0:50], "classification")
    param = parametre(model)
    assert isinstance(param, str)