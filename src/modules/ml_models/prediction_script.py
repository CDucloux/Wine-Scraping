"""Script pour réaliser les prédictions avec tous les modèles"""
from src.modules.ml_models.models import *
from src.modules.ml_models.prediction import *

for EXPLIQUEE in ("type", "unit_price"):
    if EXPLIQUEE == "type":
        MODE = "classification"
    elif EXPLIQUEE == "unit_price":
        MODE = "regression"

    X_train_n, X_test_n, y_train, y_test, _ = init(EXPLIQUEE)

    X_train = X_train_n.drop(columns=["name"])
    X_test = X_test_n.drop(columns=["name"])

    model_rf = random_forest(EXPLIQUEE, "Random Forest")
    model_boost = boosting(EXPLIQUEE, "Boosting")
    model_ridge = ridge(EXPLIQUEE, "Ridge")
    model_knn = knn(EXPLIQUEE, "K Neighbors")
    model_mlp = mlp(EXPLIQUEE, "Réseaux de neurones")
    model_sv = support_vector(EXPLIQUEE, "Support Vector")
    model_basique = basique(EXPLIQUEE)

    models = [model_rf, model_boost, model_ridge, model_knn, model_mlp, model_sv, model_basique]

    for model in models:
        model.fit(X_train, y_train)

    preds_rf = model_rf.predict(X_test).astype(str)
    preds_boost = model_boost.predict(X_test).astype(str)
    preds_ridge = model_ridge.predict(X_test).astype(str)
    preds_knn = model_knn.predict(X_test).astype(str)
    preds_mlp = model_mlp.predict(X_test).astype(str)
    preds_sv = model_sv.predict(X_test).astype(str)
    preds_basique = model_basique.predict(X_test).astype(str)

    data = {
        "name": X_test_n["name"],
        EXPLIQUEE: y_test,
        "random_forest": preds_rf,
        "boosting": preds_boost,
        "ridge": preds_ridge,
        "knn": preds_knn,
        "mlp": preds_mlp,
        "support_vector": preds_sv,
        "basique" : preds_basique
    }

    df = pl.DataFrame(data)
    df.write_csv(f"./data/tables/pred_{MODE}.csv", separator=",")
    print(f"Succès : table pred_{MODE} exportée dans le dossier data.")
