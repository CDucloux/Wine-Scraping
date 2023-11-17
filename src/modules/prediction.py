"""
Module qui réalise des prédictions à partir des optimisations de paramétre faites avec models.py
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
from models import *
from sklearn.model_selection import train_test_split
import ast
import polars as pl


def init(variable):
    """Initialise le modèle en préparant les données"""
    EXPLIQUEE = variable
    if EXPLIQUEE == "type":
        CATEGORICALS = ["cepage", "par_gouts", "service", "country"]
    elif EXPLIQUEE == "unit_price":
        CATEGORICALS = ["cepage", "par_gouts", "service", "country", "type"]

    df_dm = data_model(chemin="./data/vins.json", variable_a_predire=EXPLIQUEE)

    df = df_dm.select(
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
        "type",
    )

    df = prep_str(df, categorical_cols=CATEGORICALS)

    X = df.drop(columns=[EXPLIQUEE])
    y = df[EXPLIQUEE]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=69
    )

    df_name = df_dm.to_pandas()
    name = df_name.iloc[X_test.index]["name"]
    return X_train, X_test, y_train, y_test, name


def recup_param(choix, variable):
    """Permet de récupérer les paramètres optimaux"""
    if variable == "unit_price":
        mode = "regression"
    elif variable == "type":
        mode = "classification"

    csv = pl.read_csv("./data/result_ml_save.csv")
    csv = csv.filter(csv["Mode"] == mode)
    return ast.literal_eval(csv.filter(csv["Modèle"] == choix)["Paramètres"][0])

def prediction(model, index, X_train, y_train, X_test, y_test):
    """Réalise une prédiction sur une des données de tests"""
    model.fit(X_train, y_train)
    pred = model.predict([X_test.loc[index]])[0]
    real = y_test.loc[index]
    return pred, real


def random_forest(variable, choix, index):
    """Modèle Random Forest"""
    X_train, X_test, y_train, y_test, _ = init(variable)

    if variable == "unit_price":
        model = Pipeline(
            [
                ("imputation", SimpleImputer()),
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
                ("imputation", SimpleImputer()),
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
    pred, real = prediction(model, index, X_train, y_train, X_test, y_test)
    return pred, real


def boosting(variable, choix, index):
    """Modèle Boosting"""
    X_train, X_test, y_train, y_test, _ = init(variable)
    if variable == "unit_price":
        model = Pipeline(
            [
                ("imputation", SimpleImputer()),
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
                ("imputation", SimpleImputer()),
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
    pred, real = prediction(model, index, X_train, y_train, X_test, y_test)
    return pred, real


def ridge(variable, choix, index):
    """"Modèle Ridge"""
    X_train, X_test, y_train, y_test, _ = init(variable)
    if variable == "unit_price":
        model = Pipeline(
            [
                ("imputation", SimpleImputer()),
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
                ("imputation", SimpleImputer()),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    RidgeClassifier(
                        alpha=recup_param(choix, variable)["entrainement__alpha"]
                    ),
                ),
            ]
        )

    pred, real = prediction(model, index, X_train, y_train, X_test, y_test)
    return pred, real


def mlp(variable, choix, index):
    """Modèle MLP"""
    X_train, X_test, y_train, y_test, _ = init(variable)
    if variable == "unit_price":
        model = Pipeline(
            [
                ("imputation", SimpleImputer()),
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
                ("imputation", SimpleImputer()),
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
    pred, real = prediction(model, index, X_train, y_train, X_test, y_test)
    return pred, real


def knn(variable, choix, index):
    """Modèle KNN"""
    X_train, X_test, y_train, y_test, _ = init(variable)
    if variable == "unit_price":
        model = Pipeline(
            [
                ("imputation", SimpleImputer()),
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
                ("imputation", SimpleImputer()),
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

    pred, real = prediction(model, index, X_train, y_train, X_test, y_test)
    return pred, real


def support_vector(variable, choix, index):
    """Modèle SVM"""
    X_train, X_test, y_train, y_test, _ = init(variable)
    if variable == "unit_price":
        model = Pipeline(
            [
                ("imputation", SimpleImputer()),
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
                ("imputation", SimpleImputer()),
                ("echelle", MinMaxScaler()),
                (
                    "entrainement",
                    SVC(C=recup_param(choix, variable)["entrainement__C"]),
                ),
            ]
        )
    pred, real = prediction(model, index, X_train, y_train, X_test, y_test)
    return pred, real


def choix_utilisateur(choix_type: str, choix_selection: str, choix_vin):
    df2 = data_model(chemin="./data/vins.json", variable_a_predire="type")
    df2 = df2.to_pandas()
    index = df2[df2["name"] == choix_vin].index[0]

    if choix_type == "Regression":
        variable = "unit_price"
    elif choix_type == "Classification":
        variable = "type"

    if choix_selection == "Random Forest":
        return random_forest(variable, choix_selection, index)
    elif choix_selection == "Boosting":
        return boosting(variable, choix_selection, index)
    elif choix_selection == "Support Vector":
        return support_vector(variable, choix_selection, index)
    elif choix_selection == "Ridge":
        return ridge(variable, choix_selection, index)
    elif choix_selection == "K Neighbors":
        return knn(variable, choix_selection, index)
    elif choix_selection == "Réseaux de neurones":
        return mlp(variable, choix_selection, index)