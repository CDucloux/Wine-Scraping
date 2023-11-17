"""
Module préparait le df et l'applique aux modèles

Structure :
- Préparation : Convertit les variables qualitatives en variable binaire
- Modèles de Régressions
- Résultats
"""

from sklearn.model_selection import GridSearchCV
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
from sklearn.preprocessing import OneHotEncoder
from prettytable import PrettyTable
import numpy as np
import polars as pl
import pandas as pd
from bear_cleaner import super_pipe

# Preparation
def data_model(chemin: str, variable_a_predire: str):
    """Import le json, le transforme en dataframe, le nettoie et le prépare pour le ML"""
    df = pl.read_json(chemin)
    df = super_pipe(df)
    df = df.filter(pl.col(variable_a_predire).is_not_null())
    return df


def prep_str(df, categorical_cols: list):
    """Transforme les variables qualitatives en colonne binaire grâce à OneHotEncoder()

    >> Exemple : colonne "country": 32 pays différent
        => création de 32 colonnes binaire.
    """
    df = df.to_pandas()
    encoder = OneHotEncoder()

    encoded = encoder.fit_transform(df[categorical_cols]).toarray()

    df_encoded = pd.DataFrame(
        encoded, columns=encoder.get_feature_names_out(categorical_cols)
    )

    df = pd.concat([df.drop(columns=categorical_cols), df_encoded], axis=1)

    return df


# Modèles Régression
def model_rf(x_train, y_train, mode: str):
    """
    paramètres optimisés :
    - n_estimators
    - max_depth
    """
    if mode == "regression":
        model = Pipeline(
            [
                ("imputation", SimpleImputer()),
                ("echelle", MinMaxScaler()),
                ("entrainement", RandomForestRegressor()),
            ]
        )
    elif mode == "classification":
        model = Pipeline(
            [
                ("imputation", SimpleImputer()),
                ("echelle", MinMaxScaler()),
                ("entrainement", RandomForestClassifier()),
            ]
        )
    else:
        raise ValueError("Erreur. Utilisez 'classification' ou 'regression'.")
    cv = GridSearchCV(
        estimator=model,
        param_grid={
            "imputation__strategy": ["mean", "median", "most_frequent"],
            "entrainement__n_estimators": range(10, 50, 10),
            "entrainement__max_depth": range(1, 10, 1),
        },
        n_jobs=-1,
    )
    cv.fit(x_train, y_train)
    return cv


def model_knn(x_train, y_train, mode: str):
    """
    paramètres optimisés :
    -n_neighbors
    """
    if mode == "regression":
        model = Pipeline(
            [
                ("imputation", SimpleImputer()),
                ("echelle", MinMaxScaler()),
                ("entrainement", KNeighborsRegressor()),
            ]
        )
    elif mode == "classification":
        model = Pipeline(
            [
                ("imputation", SimpleImputer()),
                ("echelle", MinMaxScaler()),
                ("entrainement", KNeighborsClassifier()),
            ]
        )
    else:
        raise ValueError("Erreur. Utilisez 'classification' ou 'regression'.")
    cv = GridSearchCV(
        estimator=model,
        param_grid={
            "imputation__strategy": ["mean", "median", "most_frequent"],
            "entrainement__n_neighbors": range(2, 15),
        },
        n_jobs=-1,
    )
    cv.fit(x_train, y_train)
    return cv


def model_boost(x_train, y_train, mode: str):
    """
    paramètres optimisés :
    -learning_rate
    -n_estimators
    """
    if mode == "regression":
        model = Pipeline(
            [
                ("imputation", SimpleImputer()),
                ("echelle", MinMaxScaler()),
                ("entrainement", GradientBoostingRegressor()),
            ]
        )
    elif mode == "classification":
        model = Pipeline(
            [
                ("imputation", SimpleImputer()),
                ("echelle", MinMaxScaler()),
                ("entrainement", GradientBoostingClassifier()),
            ]
        )
    else:
        raise ValueError("Erreur. Utilisez 'classification' ou 'regression'.")
    
    cv = GridSearchCV(
        estimator=model,
        param_grid={
            "imputation__strategy": ["mean", "median", "most_frequent"],
            "entrainement__learning_rate": (0.005, 0.01, 0.1, 0.5),
            "entrainement__n_estimators": (50, 100, 150, 200, 400),
        },
        n_jobs=-1,
    )
    cv.fit(x_train, y_train)
    return cv


def model_mlp(x_train, y_train, mode: str):
    """
    paramètres optimisés :
    -hidden_layer_sizes
    -max_iter
    """
    if mode == "regression":
        model = Pipeline(
            [
                ("imputation", SimpleImputer()),
                ("echelle", MinMaxScaler()),
                ("entrainement", MLPRegressor()),
            ]
        )
    elif mode == "classification":
        model = Pipeline(
            [
                ("imputation", SimpleImputer()),
                ("echelle", MinMaxScaler()),
                ("entrainement", MLPClassifier()),
            ]
        )
    else:
        raise ValueError("Erreur. Utilisez 'classification' ou 'regression'.")
    cv = GridSearchCV(
        estimator=model,
        param_grid={
            "imputation__strategy": ["mean", "median", "most_frequent"],
            "entrainement__hidden_layer_sizes": [(100,), (50, 50), (60, 60)],
            "entrainement__solver": ["adam"],
            "entrainement__max_iter": [1000],
        },
        n_jobs=-1,
    )
    cv.fit(x_train, y_train)
    return cv


def model_ridge(x_train, y_train, mode):
    """
    paramètres optimisés :
    -alpha

    Ridge ajoute une pénalité à la régression linéaire standard en modifiant la fonction objectif.
    """
    if mode == "regression":
        model = Pipeline(
            [
                ("imputation", SimpleImputer()),
                ("echelle", MinMaxScaler()),
                ("entrainement", Ridge()),
            ]
        )
    elif mode == "classification":
        model = Pipeline(
            [
                ("imputation", SimpleImputer()),
                ("echelle", MinMaxScaler()),
                ("entrainement", RidgeClassifier()),
            ]
        )
    else:
        raise ValueError("Erreur. Utilisez 'classification' ou 'regression'.")
    cv = GridSearchCV(
        estimator=model,
        param_grid={
            "imputation__strategy": ["mean", "median", "most_frequent"],
            "entrainement__alpha": [2**p for p in range(-6, 7)],
        },
        n_jobs=-1,
    )
    cv.fit(x_train, y_train)
    return cv


def model_svm(x_train, y_train, mode):
    """
    paramètres optimisés :
    - C
    - epsilon
    """
    if mode == "regression":
        model = Pipeline(
            [
                ("imputation", SimpleImputer()),
                ("echelle", MinMaxScaler()),
                ("entrainement", SVR()),
            ]
        )
    elif mode == "classification":
        model = Pipeline(
            [
                ("imputation", SimpleImputer()),
                ("echelle", MinMaxScaler()),
                ("entrainement", SVC()),
            ]
        )
    else:
        raise ValueError("Erreur. Utilisez 'classification' ou 'regression'.")
    cv = GridSearchCV(
        estimator=model,
        param_grid={
            "imputation__strategy": ["mean", "median", "most_frequent"],
            "entrainement__C": np.logspace(-4, 16, 15, base=2)
        },
        n_jobs=-1,
    )
    cv.fit(x_train, y_train)
    return cv


def train_model(x_train, y_train, mode):
    """Fonction entrainant tous les modèles"""
    return {
        "model_knn": model_knn(x_train, y_train, mode),
        "model_rf": model_rf(x_train, y_train, mode),
        "model_boost": model_boost(x_train, y_train, mode),
        "model_ridge": model_ridge(x_train, y_train, mode),
        "model_svm": model_svm(x_train, y_train, mode),
        "model_mlp": model_mlp(x_train, y_train, mode),
    }


# Résultats notebook
def model_result(**kwargs):
    """Permet de faire un résumer de(s) résultat(s) de(s) modèle(s)
    >> model_result(nom_du_mode = model_entrainé)"""
    table = PrettyTable(["Modèle", "Score", "SD"])
    for nom, model in kwargs.items():
        indice_meilleur = model.cv_results_["rank_test_score"].argmin()
        table.add_row(
            [
                nom,
                round(model.cv_results_["mean_test_score"][indice_meilleur], 3),
                round(model.cv_results_["std_test_score"][indice_meilleur], 3),
            ]
        )
    return print(table)


def model_param(modele, *args):
    """Permet de connaître les meilleurs paramètres pour un modèle
    >> model_param(model_entrainé, "entrainement__nom_du_parametre")"""
    indice_meilleur = modele.cv_results_["rank_test_score"].argmin()
    table2 = PrettyTable(["Parameter", "Value"])
    for key in args:
        table2.add_row([key, modele.cv_results_["params"][indice_meilleur][key]])
    return print(table2)


# Résultats
def score(model):
    """Retourne le score"""
    indice_meilleur = model.cv_results_["rank_test_score"].argmin()
    return round(model.cv_results_["mean_test_score"][indice_meilleur], 3)


def ecart_type(model):
    """Retourne l'ecart-type"""
    indice_meilleur = model.cv_results_["rank_test_score"].argmin()
    return round(model.cv_results_["std_test_score"][indice_meilleur], 3)


def parametre(model):
    """Retourne les meilleurs paramètres"""
    indice_meilleur = model.cv_results_["rank_test_score"].argmin()
    return str(model.cv_results_["params"][indice_meilleur])


def stockage_result_csv(model, mode):
    """Stock les résultats dans un CSV"""
    ml = {
        "Modèle": [
            "Random Forest",
            "K Neighbors",
            "Réseaux de neurones",
            "Boosting",
            "Ridge",
            "Support Vector",
        ],
        "Score": [
            score(model["model_rf"]),
            score(model["model_knn"]),
            score(model["model_mlp"]),
            score(model["model_boost"]),
            score(model["model_ridge"]),
            score(model["model_svm"]),
        ],
        "Ecart-Type": [
            ecart_type(model["model_rf"]),
            ecart_type(model["model_knn"]),
            ecart_type(model["model_mlp"]),
            ecart_type(model["model_boost"]),
            ecart_type(model["model_ridge"]),
            ecart_type(model["model_svm"]),
        ],
        "Paramètres": [
            parametre(model["model_rf"]),
            parametre(model["model_knn"]),
            parametre(model["model_mlp"]),
            parametre(model["model_boost"]),
            parametre(model["model_ridge"]),
            parametre(model["model_svm"]),
        ],
        "Mode" :[
            mode,
            mode,
            mode,
            mode,
            mode,
            mode
        ]
    }
    ml = pl.DataFrame(ml)
    ml.write_csv("./data/result_ml.csv", separator=",")
    return print("C'est bon ça a marché")
