"""
Module de préparation des modèles
=================================

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
import numpy as np
import polars as pl
import pandas as pd
from src.modules.bear_cleaner import super_pipe


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
        return_train_score=True,
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
        return_train_score=True,
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
        return_train_score=True,
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
        return_train_score=True,
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
        return_train_score=True,
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
            "entrainement__C": np.logspace(-4, 16, 15, base=2),
        },
        n_jobs=-1,
        return_train_score=True,
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


# Résultats
def score_test(model):
    """Retourne le score de test"""
    indice_meilleur = model.cv_results_["rank_test_score"].argmin()
    return round(model.cv_results_["mean_test_score"][indice_meilleur], 3)


def score_entrainement(model):
    """Retourne le score d'entrainement"""
    indice_meilleur = model.cv_results_["rank_test_score"].argmin()
    return round(model.cv_results_["mean_train_score"][indice_meilleur], 3)


def ecart_type_test(model):
    """Retourne l'ecart-type"""
    indice_meilleur = model.cv_results_["rank_test_score"].argmin()
    return round(model.cv_results_["std_test_score"][indice_meilleur], 3)


def ecart_type_train(model):
    """Retourne l'ecart-type"""
    indice_meilleur = model.cv_results_["rank_test_score"].argmin()
    return round(model.cv_results_["std_train_score"][indice_meilleur], 3)


def parametre(model):
    """Retourne les meilleurs paramètres"""
    indice_meilleur = model.cv_results_["rank_test_score"].argmin()
    return str(model.cv_results_["params"][indice_meilleur])
