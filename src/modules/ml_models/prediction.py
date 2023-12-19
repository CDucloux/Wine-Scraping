"""
Module qui réalise des prédictions à partir des optimisations de paramètres faits avec models.py
"""
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.svm import SVR, SVC
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from src.modules.ml_models.models import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

import ast
import polars as pl


def init(variable, chemin = "./data/vins.json"):
    """Initialise le modèle en préparant les données"""
    EXPLIQUEE = variable
    if EXPLIQUEE == "type":
        CATEGORICALS = ["cepage", "par_gouts", "service", "country"]
    elif EXPLIQUEE == "unit_price":
        CATEGORICALS = ["cepage", "par_gouts", "service", "country", "type"]

    df_dm = data_model(chemin = chemin, variable_a_predire = EXPLIQUEE)
 
    df = df_dm.select(
        "name",
        "capacity",
        "unit_price",
        "millesime",
        "cepage",
        "par_gouts",
        "service",
        "avg_temp",
        "conservation_date",
        "bio",
        "customer_fav",
        "is_new",
        "top_100",
        "destock",
        "sulphite_free",
        "alcohol_volume",
        "country",
        "bubbles",
        "wine_note",
        "nb_reviews",
        "conservation_time",
        "cru",
        "type",
    )

    df = prep_str(df, categorical_cols=CATEGORICALS)

    X = df.drop(columns=[EXPLIQUEE])
    y = df[EXPLIQUEE]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=69
    )
    return X_train, X_test, y_train, y_test, df


def recup_param(choix, variable):
    """Permet de récupérer les paramètres optimaux"""
    if variable == "unit_price":
        mode = "regression"
        csv = pl.read_csv("./data/tables/result_ml_regression.csv")
    elif variable == "type":
        mode = "classification"
        csv = pl.read_csv("./data/tables/result_ml_classification.csv")

    csv = csv.filter(csv["Mode"] == mode)
    return ast.literal_eval(csv.filter(csv["Modèle"] == choix)["Paramètres"][0])


def random_forest(variable, choix):
    """Modèle Random Forest"""
    if variable == "unit_price":
        model = Pipeline(
            [
                (
                    "imputation",
                    SimpleImputer(
                        strategy=recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    RandomForestRegressor(
                        max_depth=recup_param(choix, variable)[
                            "entrainement__max_depth"
                        ],
                        n_estimators=recup_param(choix, variable)[
                            "entrainement__n_estimators"
                        ],
                    ),
                ),
            ]
        )
    elif variable == "type":
        model = Pipeline(
            [
                (
                    "imputation",
                    SimpleImputer(
                        strategy=recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    RandomForestClassifier(
                        max_depth=recup_param(choix, variable)[
                            "entrainement__max_depth"
                        ],
                        n_estimators=recup_param(choix, variable)[
                            "entrainement__n_estimators"
                        ],
                    ),
                ),
            ]
        )
    return model


def boosting(variable, choix):
    """Modèle Boosting"""
    if variable == "unit_price":
        model = Pipeline(
            [
                (
                    "imputation",
                    SimpleImputer(
                        strategy=recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    GradientBoostingRegressor(
                        learning_rate=recup_param(choix, variable)[
                            "entrainement__learning_rate"
                        ],
                        n_estimators=recup_param(choix, variable)[
                            "entrainement__n_estimators"
                        ],
                    ),
                ),
            ]
        )
    elif variable == "type":
        model = Pipeline(
            [
                (
                    "imputation",
                    SimpleImputer(
                        strategy=recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    GradientBoostingClassifier(
                        learning_rate=recup_param(choix, variable)[
                            "entrainement__learning_rate"
                        ],
                        n_estimators=recup_param(choix, variable)[
                            "entrainement__n_estimators"
                        ],
                    ),
                ),
            ]
        )
    return model


def ridge(variable, choix):
    """ "Modèle Ridge"""
    if variable == "unit_price":
        model = Pipeline(
            [
                (
                    "imputation",
                    SimpleImputer(
                        strategy=recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    Ridge(alpha=recup_param(choix, variable)["entrainement__alpha"]),
                ),
            ]
        )
    elif variable == "type":
        model = Pipeline(
            [
                (
                    "imputation",
                    SimpleImputer(
                        strategy=recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    RidgeClassifier(
                        alpha=recup_param(choix, variable)["entrainement__alpha"]
                    ),
                ),
            ]
        )
    return model


def mlp(variable, choix):
    """Modèle MLP"""
    if variable == "unit_price":
        model = Pipeline(
            [
                (
                    "imputation",
                    SimpleImputer(
                        strategy=recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    MLPRegressor(
                        hidden_layer_sizes=recup_param(choix, variable)[
                            "entrainement__hidden_layer_sizes"
                        ],
                        solver=recup_param(choix, variable)["entrainement__solver"],
                        max_iter=recup_param(choix, variable)["entrainement__max_iter"],
                    ),
                ),
            ]
        )
    elif variable == "type":
        model = Pipeline(
            [
                (
                    "imputation",
                    SimpleImputer(
                        strategy=recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    MLPClassifier(
                        hidden_layer_sizes=recup_param(choix, variable)[
                            "entrainement__hidden_layer_sizes"
                        ],
                        solver=recup_param(choix, variable)["entrainement__solver"],
                        max_iter=recup_param(choix, variable)["entrainement__max_iter"],
                    ),
                ),
            ]
        )
    return model


def knn(variable, choix):
    """Modèle KNN"""
    if variable == "unit_price":
        model = Pipeline(
            [
                (
                    "imputation",
                    SimpleImputer(
                        strategy=recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    KNeighborsRegressor(
                        n_neighbors=recup_param(choix, variable)[
                            "entrainement__n_neighbors"
                        ]
                    ),
                ),
            ]
        )
    elif variable == "type":
        model = Pipeline(
            [
                (
                    "imputation",
                    SimpleImputer(
                        strategy=recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    KNeighborsClassifier(
                        n_neighbors=recup_param(choix, variable)[
                            "entrainement__n_neighbors"
                        ]
                    ),
                ),
            ]
        )
    return model


def support_vector(variable, choix):
    """Modèle SVM"""
    if variable == "unit_price":
        model = Pipeline(
            [
                (
                    "imputation",
                    SimpleImputer(
                        strategy=recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    SVR(C=recup_param(choix, variable)["entrainement__C"]),
                ),
            ]
        )
    elif variable == "type":
        model = Pipeline(
            [
                (
                    "imputation",
                    SimpleImputer(
                        strategy=recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    SVC(C=recup_param(choix, variable)["entrainement__C"]),
                ),
            ]
        )
    return model


def performance(variable):
    """Sert de contrôle ..."""
    erreur_test = list()
    X_train_n, X_test_n, y_train, y_test, _ = init(variable)
    
    X_train = X_train_n.drop(columns=["name"])
    X_test = X_test_n.drop(columns=["name"])

    models = list()
    model_functions = [
        ("Random Forest", random_forest),
        ("K Neighbors", knn),
        ("Réseaux de neurones", mlp),
        ("Boosting", boosting),
        ("Ridge", ridge),
        ("Support Vector", support_vector),
    ]
    for model_name, model_function in model_functions:
        model = model_function(variable, model_name)
        model = model.fit(X_train, y_train)
        models.append(model)

    for model in models:
        y_pred = model.predict(X_test)
        if variable == "type":
            erreur_test.append(accuracy_score(y_test, y_pred))
        elif variable == "unit_price":
            erreur_test.append(mean_absolute_error(y_test, y_pred))
    return erreur_test


def stockage_result_csv(model, mode: str):
    """Stocke les résultats dans un CSV"""
    if mode == "regression":
        variable = "unit_price"
    elif mode == "classification":
        variable = "type"
    ml = {
        "Modèle": [
            "Random Forest",
            "K Neighbors",
            "Réseaux de neurones",
            "Boosting",
            "Ridge",
            "Support Vector",
        ],
        "Score Test": [
            score_test(model["model_rf"]),
            score_test(model["model_knn"]),
            score_test(model["model_mlp"]),
            score_test(model["model_boost"]),
            score_test(model["model_ridge"]),
            score_test(model["model_svm"]),
        ],
        "Score Entrainement": [
            score_entrainement(model["model_rf"]),
            score_entrainement(model["model_knn"]),
            score_entrainement(model["model_mlp"]),
            score_entrainement(model["model_boost"]),
            score_entrainement(model["model_ridge"]),
            score_entrainement(model["model_svm"]),
        ],
        "Ecart-Type Test": [
            ecart_type_test(model["model_rf"]),
            ecart_type_test(model["model_knn"]),
            ecart_type_test(model["model_mlp"]),
            ecart_type_test(model["model_boost"]),
            ecart_type_test(model["model_ridge"]),
            ecart_type_test(model["model_svm"]),
        ],
        "Ecart-Type Train": [
            ecart_type_train(model["model_rf"]),
            ecart_type_train(model["model_knn"]),
            ecart_type_train(model["model_mlp"]),
            ecart_type_train(model["model_boost"]),
            ecart_type_train(model["model_ridge"]),
            ecart_type_train(model["model_svm"]),
        ],
        "Paramètres": [
            parametre(model["model_rf"]),
            parametre(model["model_knn"]),
            parametre(model["model_mlp"]),
            parametre(model["model_boost"]),
            parametre(model["model_ridge"]),
            parametre(model["model_svm"]),
        ],
        "Score Test data": [
            performance(variable)[0],
            performance(variable)[1],
            performance(variable)[2],
            performance(variable)[3],
            performance(variable)[4],
            performance(variable)[5],
        ],
        "Mode": [mode, mode, mode, mode, mode, mode],
    }
    ml = pl.DataFrame(ml)
    ml.write_csv(f"./data/tables/result_ml_{mode}.csv", separator=",")
    return print("Succès")
