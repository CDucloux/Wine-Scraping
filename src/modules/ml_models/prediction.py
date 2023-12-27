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
import warnings
from enum import Enum

warnings.filterwarnings("ignore")

class targets(Enum):
    """Enumération modélisant les 2 variables à prédire possibles."""

    PRICE = "unit_price"
    TYPE = "type"


def init(
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """`init`: Initialise les données et les prépare au Machine Learning.
    Effectue un Train/Test split (80%/20%) et renvoie un tuple contenant :

    - Les features d'entrainement
    - Les features de test
    - La target d'entrainement
    - La target de test
    - Le DataFrame initial

    ---------
    `Parameters`
    --------- ::

        target (str): # La variable à prédire

    `Returns`
    --------- ::

        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]

    `Example(s)`
    ---------

    >>> X_train, X_test, y_train, y_test, df = init(targets.PRICE.value)
    ... X_train.shape, X_test.shape, y_train.shape, y_test.shape, df.shape
    ... ((3204, 253), (802, 253), (3204,), (802,), (4006, 254))
    """
    EXPLIQUEE = target
    if EXPLIQUEE == targets.TYPE.value:
        CATEGORICALS = ["cepage", "par_gouts", "service", "country"]
    elif EXPLIQUEE == targets.PRICE.value:
        CATEGORICALS = ["cepage", "par_gouts", "service", "country", "type"]

    df_dm = data_model(path="./data/vins.json", target=EXPLIQUEE)

    df = df_dm.select(
        # "name",
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


def _recup_param(choix: str, target: str) -> dict:
    """`_recup_param`: Permet de récupérer les paramètres optimaux dans le CSV de résultats du Machine Learning.

    ---------
    `Parameters`
    --------- ::

        choix (str): # Le choix du modèle de Machine Learning
        target (str): # La variable à prédire

    `Returns`
    --------- ::

        dict

    `Example(s)`
    ---------

    - Exemple 1 : `Random Forest` et `classification`
    >>> _recup_param("Random Forest", "type")
    ... {'entrainement__max_depth': 9,
    ...  'entrainement__n_estimators': 30,
    ...  'imputation__strategy': 'median'}

    - Exemple 2 : `Boosting` et `regression`

    >>> _recup_param("Boosting", "unit_price")
    ... {'entrainement__learning_rate': 0.1,
    ... 'entrainement__n_estimators': 150,
    ... 'imputation__strategy': 'most_frequent'}
    """
    if target == "unit_price":
        csv = pl.read_csv("./data/tables/result_ml_regression.csv")
    elif target == "type":
        csv = pl.read_csv("./data/tables/result_ml_classification.csv")

    return ast.literal_eval(csv.filter(csv["Modèle"] == choix)["Paramètres"][0])


def random_forest(variable: str, choix: str) -> Pipeline:
    """`random_forest`: permet de préparer le pipeline du modèle Random Forest en récupérant les paramètres optimaux avec `_recup_param`

    ---------
    `Parameters`
    --------- ::

        variable (str): # La variable à prédire
        choix (str): # Le choix du modèle de Machine Learning

    `Returns`
    --------- ::

        Pipeline

    `Example(s)`
    ---------

    - Exemple d'utilisation : `classification`
    >>> model = random_forest("type", "Random Forest")
    >>> model.fit(X_train, y_train)
    >>> model.predict(X_test)
    """
    if variable == "unit_price":
        model = Pipeline(
            [
                (
                    "imputation",
                    SimpleImputer(
                        strategy=_recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    RandomForestRegressor(
                        max_depth=_recup_param(choix, variable)[
                            "entrainement__max_depth"
                        ],
                        n_estimators=_recup_param(choix, variable)[
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
                        strategy=_recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    RandomForestClassifier(
                        max_depth=_recup_param(choix, variable)[
                            "entrainement__max_depth"
                        ],
                        n_estimators=_recup_param(choix, variable)[
                            "entrainement__n_estimators"
                        ],
                    ),
                ),
            ]
        )
    return model


def boosting(variable: str, choix: str) -> Pipeline:
    """`boosting`: permet de préparer le pipeline du modèle Boosting en récupérant les paramètres optimaux avec `_recup_param`

    ---------
    `Parameters`
    --------- ::

        variable (str): # La variable à prédire
        choix (str): # Le choix du modèle de Machine Learning

    `Returns`
    --------- ::

        Pipeline

    `Example(s)`
    ---------

    - Exemple d'utilisation : `classification`
    >>> model = boosting("type", "Boosting")
    >>> model.fit(X_train, y_train)
    >>> model.predict(X_test)
    """
    if variable == "unit_price":
        model = Pipeline(
            [
                (
                    "imputation",
                    SimpleImputer(
                        strategy=_recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    GradientBoostingRegressor(
                        learning_rate=_recup_param(choix, variable)[
                            "entrainement__learning_rate"
                        ],
                        n_estimators=_recup_param(choix, variable)[
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
                        strategy=_recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    GradientBoostingClassifier(
                        learning_rate=_recup_param(choix, variable)[
                            "entrainement__learning_rate"
                        ],
                        n_estimators=_recup_param(choix, variable)[
                            "entrainement__n_estimators"
                        ],
                    ),
                ),
            ]
        )
    return model


def ridge(variable, choix) -> Pipeline:
    """`ridge`: permet de préparer le pipeline du modèle Ridge en récupérant les paramètres optimaux avec `_recup_param`

    ---------
    `Parameters`
    --------- ::

        variable (str): # La variable à prédire
        choix (str): # Le choix du modèle de Machine Learning

    `Returns`
    --------- ::

        Pipeline

    `Example(s)`
    ---------

    - Exemple d'utilisation : `classification`
    >>> model = ridge("type", "Ridge")
    >>> model.fit(X_train, y_train)
    >>> model.predict(X_test)
    """
    if variable == "unit_price":
        model = Pipeline(
            [
                (
                    "imputation",
                    SimpleImputer(
                        strategy=_recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    Ridge(alpha=_recup_param(choix, variable)["entrainement__alpha"]),
                ),
            ]
        )
    elif variable == "type":
        model = Pipeline(
            [
                (
                    "imputation",
                    SimpleImputer(
                        strategy=_recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    RidgeClassifier(
                        alpha=_recup_param(choix, variable)["entrainement__alpha"]
                    ),
                ),
            ]
        )
    return model


def mlp(variable, choix) -> Pipeline:
    """`mlp`: permet de préparer le pipeline du modèle Réseaux de neurones en récupérant les paramètres optimaux avec `_recup_param`

    ---------
    `Parameters`
    --------- ::

        variable (str): # La variable à prédire
        choix (str): # Le choix du modèle de Machine Learning

    `Returns`
    --------- ::

        Pipeline

    `Example(s)`
    ---------

    - Exemple d'utilisation : `classification`
    >>> model = mlp("type", "Réseaux de neurones")
    >>> model.fit(X_train, y_train)
    >>> model.predict(X_test)
    """
    if variable == "unit_price":
        model = Pipeline(
            [
                (
                    "imputation",
                    SimpleImputer(
                        strategy=_recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    MLPRegressor(
                        hidden_layer_sizes=_recup_param(choix, variable)[
                            "entrainement__hidden_layer_sizes"
                        ],
                        solver=_recup_param(choix, variable)["entrainement__solver"],
                        max_iter=_recup_param(choix, variable)[
                            "entrainement__max_iter"
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
                        strategy=_recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    MLPClassifier(
                        hidden_layer_sizes=_recup_param(choix, variable)[
                            "entrainement__hidden_layer_sizes"
                        ],
                        solver=_recup_param(choix, variable)["entrainement__solver"],
                        max_iter=_recup_param(choix, variable)[
                            "entrainement__max_iter"
                        ],
                    ),
                ),
            ]
        )
    return model


def knn(variable, choix) -> Pipeline:
    """`knn`: permet de préparer le pipeline du modèle K Neighbors en récupérant les paramètres optimaux avec `_recup_param`

    ---------
    `Parameters`
    --------- ::

        variable (str): # La variable à prédire
        choix (str): # Le choix du modèle de Machine Learning

    `Returns`
    --------- ::

        Pipeline

    `Example(s)`
    ---------

    - Exemple d'utilisation : `classification`
    >>> model = knn("type", "K Neighbors")
    >>> model.fit(X_train, y_train)
    >>> model.predict(X_test)
    """
    if variable == "unit_price":
        model = Pipeline(
            [
                (
                    "imputation",
                    SimpleImputer(
                        strategy=_recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    KNeighborsRegressor(
                        n_neighbors=_recup_param(choix, variable)[
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
                        strategy=_recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    KNeighborsClassifier(
                        n_neighbors=_recup_param(choix, variable)[
                            "entrainement__n_neighbors"
                        ]
                    ),
                ),
            ]
        )
    return model


def support_vector(variable, choix) -> Pipeline:
    """`support_vector`: permet de préparer le pipeline du modèle Support Vector en récupérant les paramètres optimaux avec `_recup_param`

    ---------
    `Parameters`
    --------- ::

        variable (str): # La variable à prédire
        choix (str): # Le choix du modèle de Machine Learning

    `Returns`
    --------- ::

        Pipeline

    `Example(s)`
    ---------

    - Exemple d'utilisation : `classification`
    >>> model = support_vector("type", "Support Vector")
    >>> model.fit(X_train, y_train)
    >>> model.predict(X_test)
    """
    if variable == "unit_price":
        model = Pipeline(
            [
                (
                    "imputation",
                    SimpleImputer(
                        strategy=_recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    SVR(C=_recup_param(choix, variable)["entrainement__C"]),
                ),
            ]
        )
    elif variable == "type":
        model = Pipeline(
            [
                (
                    "imputation",
                    SimpleImputer(
                        strategy=_recup_param(choix, variable)["imputation__strategy"]
                    ),
                ),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    SVC(C=_recup_param(choix, variable)["entrainement__C"]),
                ),
            ]
        )
    return model


def performance(target: str) -> list:
    """`performance`: permet de mesurer la performance des modèles.
    
    Le résultat se retrouve dans `result_ml_classification.csv`/ `result_ml_regression.csv`. \\
    L'intérêt pour le développeur est que ce résultat soit proche de ce qui s'affiche dans l'application. \\
    Sinon c'est qu'il y a un problème soit dans `prediction.py` soit dans `st_tables.py.
    
    Metrics : 
    - Classification : accuracy_score
    - Regression : mean_absolute_error

    ---------
    `Parameters`
    --------- ::

        target (str): # La variable à prédire

    `Returns`
    --------- ::

        list

    `Example(s)`
    ---------

    - Exemple d'utilisation : `classification`
    >>>
    """
    erreur_test = list()
    X_train, X_test, y_train, y_test, _ = init(target)

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
        model = model_function(target, model_name)
        model = model.fit(X_train, y_train)
        models.append(model)

    for model in models:
        y_pred = model.predict(X_test)
        if target == "type":
            erreur_test.append(accuracy_score(y_test, y_pred))
        elif target == "unit_price":
            erreur_test.append(mean_absolute_error(y_test, y_pred))
    return erreur_test


def stockage_result_csv(model, mode: str):
    """`stockage_result_csv`: créer un CSV avec les scores, écarts-types et performances des modèles
    
    ---------
    `Parameters`
    --------- ::
    
        model: modèles entrainer (par exemple avec `train_model`)
        mode (str): Type de prédication : régression ou classification

    `Returns`
    --------- ::

        "Succès": csv créée

    `Example(s)`
    ---------

    - Exemple d'utilisation : 
    >>> stockage_result_csv(models, "classification")
    ... "Succès"
    """
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
