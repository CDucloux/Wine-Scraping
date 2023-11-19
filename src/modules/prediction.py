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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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


def random_forest(variable, choix, index=None):
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
    if index is not None:
        pred, real = prediction(model, index, X_train, y_train, X_test, y_test)
    else:
        pred = None
        real = None
    return pred, real, model


def boosting(variable, choix, index=None):
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
    if index is not None:
        pred, real = prediction(model, index, X_train, y_train, X_test, y_test)
    else:
        pred = None
        real = None
    return pred, real, model


def ridge(variable, choix, index=None):
    """ "Modèle Ridge"""
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
    if index is not None:
        pred, real = prediction(model, index, X_train, y_train, X_test, y_test)
    else:
        pred = None
        real = None
    return pred, real, model


def mlp(variable, choix, index=None):
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
    if index is not None:
        pred, real = prediction(model, index, X_train, y_train, X_test, y_test)
    else:
        pred = None
        real = None
    return pred, real, model


def knn(variable, choix, index=None):
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
    if index is not None:
        pred, real = prediction(model, index, X_train, y_train, X_test, y_test)
    else:
        pred = None
        real = None
    return pred, real, model


def support_vector(variable, choix, index=None):
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
    if index is not None:
        pred, real = prediction(model, index, X_train, y_train, X_test, y_test)
    else:
        pred = None
        real = None
    return pred, real, model


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


def performance(variable):
    erreur_test = []
    X_train, X_test, y_train, y_test, _ = init(variable)

    models = []
    model_functions = [
        ("Random Forest", random_forest),
        ("K Neighbors", knn),
        ("Réseaux de neurones", mlp),
        ("Boosting", boosting),
        ("Ridge", ridge),
        ("Support Vector", support_vector),
    ]
    for model_name, model_function in model_functions:
        _, _, model = model_function(variable, model_name)
        model = model.fit(X_train, y_train)
        models.append(model)

    for model in models:
        y_pred = model.predict(X_test)
        if variable == "type":
            erreur_test.append(accuracy_score(y_test, y_pred))
        elif variable == "unit_price":
            erreur_test.append(mean_absolute_error(y_test, y_pred))
    return erreur_test


def stockage_result_csv(model, mode):
    """Stock les résultats dans un CSV"""
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
    ml.write_csv(f"./data/result_ml_{mode}.csv", separator=",")
    return print("C'est bon ça a marché")
