"""Script pour optimiser les paramètres des modèles."""
from src.modules.ml_models.models import train_model
from src.modules.ml_models.prediction import (
    init,
    stockage_result_csv
)

for EXPLIQUEE in ("type", "unit_price"):
    X_train_n, X_test_n, y_train, y_test, _ = init(EXPLIQUEE)

    X_train = X_train_n.drop(columns=["name"])
    X_test = X_test_n.drop(columns=["name"])

    if EXPLIQUEE == "type":
        MODE = "classification"
    elif EXPLIQUEE == "unit_price":
        MODE = "regression"

    print(f"Mode sélectionné : {MODE}")

    models = train_model(X_train, y_train, MODE)

    stockage_result_csv(models, MODE)
