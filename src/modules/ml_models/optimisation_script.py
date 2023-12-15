"""Script pour optimiser les paramètres des modèles."""
from src.modules.ml_models.models import *
from src.modules.ml_models.prediction import *

EXPLIQUEE = "type" #type or unit_price

X_train, X_test, y_train, y_test, _ = init(EXPLIQUEE)

if EXPLIQUEE == "type":
    MODE = "classification"
elif EXPLIQUEE == "unit_price":
    MODE = "regression"

models = train_model(X_train, y_train, MODE)

stockage_result_csv(models, MODE)