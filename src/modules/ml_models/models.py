"""
Module de préparation des modèles
=================================

Structure :
- Préparation : Convertit les variables qualitatives en variable binaire
- Modèles de régression et classification 
- Résultats
"""

from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier  # type: ignore
from sklearn.ensemble import (  # type: ignore
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.neural_network import MLPRegressor, MLPClassifier  # type: ignore
from sklearn.linear_model import Ridge, RidgeClassifier  # type: ignore
from sklearn.svm import SVR, SVC  # type: ignore
from sklearn.impute import SimpleImputer  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder  # type: ignore
import numpy as np
import polars as pl
import pandas as pd
from src.modules.utils import model_name
from src.modules.bear_cleaner import super_pipe  # type: ignore

import warnings

warnings.filterwarnings("ignore")


def data_model(chemin: str, variable_a_predire: str) -> pl.DataFrame:
    """Importe le json, le transforme en dataframe, le nettoie et le prépare pour le ML"""
    df_brut = pl.read_json(chemin)
    df = super_pipe(df_brut)
    df = df.filter(pl.col(variable_a_predire).is_not_null())
    return df


def prep_str(df: pl.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """`prep_str`: Transforme les variables qualitatives en colonnes binaires grâce au `OneHotEncoder()`.
    Renvoie un DataFrame avec un nombre important de variables numériques binaires.

    ---------
    `Parameters`
    --------- ::

        df (pl.DataFrame): # Le DataFrame initial
        categorical_cols (list): # Une liste des variables catégorielles

    `Returns`
    --------- ::

        pd.DataFrame

    `Example(s)`
    ---------

    >>> prep_str()
    ... #_test_return_"""
    df_pd = df.to_pandas()
    encoder = OneHotEncoder()

    encoded = encoder.fit_transform(df_pd[categorical_cols]).toarray()

    df_encoded = pd.DataFrame(
        encoded, columns=encoder.get_feature_names_out(categorical_cols)
    )

    df_pd = pd.concat([df_pd.drop(columns=categorical_cols), df_encoded], axis=1)

    return df_pd


@model_name
def model_rf(x_train: pd.DataFrame, y_train: pd.Series, mode: str) -> GridSearchCV:
    """`model_rf`: Effectue une recherche exhaustive (Cross-Validation) des meilleurs paramètres
    en utilisant une Random Forest. Les paramètres optimisés sont :

    - n_estimators
    - max_depth

    ---------
    `Parameters`
    --------- ::

        x_train (pd.DataFrame): # L'ensemble d'entrainement
        y_train (pd.Series): # La variable à prédire
        mode (str): # regression | classification

    `Raises`
    --------- ::

        ValueError: # Une erreur est levée quand le mode est invalide

    `Returns`
    --------- ::

        GridSearchCV

    `Example(s)`
    ---------

    >>> model_rf()
    ... #_test_return_"""
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


@model_name
def model_knn(x_train: pd.DataFrame, y_train: pd.Series, mode: str) -> GridSearchCV:
    """`model_knn`: Effectue une recherche exhaustive (Cross-Validation) des meilleurs paramètres
    en utilisant un KNN. Les paramètres optimisés sont :

    - n_neighbors

    ---------
    `Parameters`
    --------- ::

        x_train (pd.DataFrame): # L'ensemble d'entrainement
        y_train (pd.Series): # La variable à prédire
        mode (str): # regression | classification

    `Raises`
    --------- ::

        ValueError: # Une erreur est levée quand le mode est invalide

    `Returns`
    --------- ::

        GridSearchCV

    `Example(s)`
    ---------

    >>> model_knn()
    ... #_test_return_"""
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


@model_name
def model_boost(x_train: pd.DataFrame, y_train: pd.Series, mode: str) -> GridSearchCV:
    """`model_boost`: Effectue une recherche exhaustive (Cross-Validation) des meilleurs paramètres
    en utilisant un Gradient Boosting. Les paramètres optimisés sont :

    - learning_rate
    - n_estimators

    ---------
    `Parameters`
    --------- ::

        x_train (pd.DataFrame): # L'ensemble d'entrainement
        y_train (pd.Series): # La variable à prédire
        mode (str): # regression | classification

    `Raises`
    --------- ::

        ValueError: # Une erreur est levée quand le mode est invalide

    `Returns`
    --------- ::

        GridSearchCV

    `Example(s)`
    ---------

    >>> model_boost()
    ... #_test_return_"""
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


@model_name
def model_mlp(x_train: pd.DataFrame, y_train: pd.Series, mode: str) -> GridSearchCV:
    """`model_mlp`: Effectue une recherche exhaustive (Cross-Validation) des meilleurs paramètres
    en utilisant un Gradient Boosting. Les paramètres optimisés sont :

    - hidden_layer_sizes
    - max_iter

    ---------
    `Parameters`
    --------- ::

        x_train (pd.DataFrame): # L'ensemble d'entrainement
        y_train (pd.Series): # La variable à prédire
        mode (str): # regression | classification

    `Raises`
    --------- ::

        ValueError: # Une erreur est levée quand le mode est invalide

    `Returns`
    --------- ::

        GridSearchCV

    `Example(s)`
    ---------

    >>> model_mlp()
    ... #_test_return_"""
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


@model_name
def model_ridge(x_train: pd.DataFrame, y_train: pd.Series, mode: str) -> GridSearchCV:
    """`model_ridge`: Effectue une recherche exhaustive (Cross-Validation) des meilleurs paramètres
    en utilisant un modèle Ridge. Les paramètres optimisés sont :

    - alpha

    Note : Ridge ajoute une pénalité à la régression linéaire standard en modifiant la fonction d'objectif.

    ---------
    `Parameters`
    --------- ::

        x_train (pd.DataFrame): # L'ensemble d'entrainement
        y_train (pd.Series): # La variable à prédire
        mode (str): # regression | classification

    `Raises`
    --------- ::

        ValueError: # Une erreur est levée quand le mode est invalide

    `Returns`
    --------- ::

        GridSearchCV

    `Example(s)`
    ---------

    >>> model_ridge()
    ... #_test_return_"""
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


@model_name
def model_svm(x_train: pd.DataFrame, y_train: pd.Series, mode: str) -> GridSearchCV:
    """`model_svm`: Effectue une recherche exhaustive (Cross-Validation) des meilleurs paramètres
    en utilisant un SVM. Les paramètres optimisés sont :

    - C
    - epsilon

    ---------
    `Parameters`
    --------- ::

        x_train (pd.DataFrame): # L'ensemble d'entrainement
        y_train (pd.Series): # La variable à prédire
        mode (str): # regression | classification

    `Raises`
    --------- ::

        ValueError: # Une erreur est levée quand le mode est invalide

    `Returns`
    --------- ::

        GridSearchCV

    `Example(s)`
    ---------

    >>> model_svm()
    ... #_test_return_"""
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


def train_model(
    x_train: pd.DataFrame, y_train: pd.Series, mode: str
) -> dict[str, GridSearchCV]:
    """`train_model`: Fonction entrainant tous les modèles.
    Renvoie un dictionnaire permettant d'accéder à chaque modèle et ses hyperparamètres.

    ---------
    `Parameters`
    --------- ::

        x_train (pd.DataFrame): # L'ensemble d'entrainement
        y_train (pd.Series): # La variable à prédire
        mode (str): # regression | classification

    `Returns`
    --------- ::

        dict[str, GridSearchCV]

    `Example(s)`
    ---------

    >>> train_model()
    ... #_test_return_"""
    return {
        "model_knn": model_knn(x_train, y_train, mode),
        "model_rf": model_rf(x_train, y_train, mode),
        "model_boost": model_boost(x_train, y_train, mode),
        "model_ridge": model_ridge(x_train, y_train, mode),
        "model_svm": model_svm(x_train, y_train, mode),
        "model_mlp": model_mlp(x_train, y_train, mode),
    }


def score_test(model: GridSearchCV) -> np.float64:
    """Retourne le score de test du meilleur modèle."""
    indice_meilleur = model.cv_results_["rank_test_score"].argmin()
    return round(model.cv_results_["mean_test_score"][indice_meilleur], 3)


def score_entrainement(model: GridSearchCV) -> np.float64:
    """Retourne le score d'entrainement du meilleur modèle."""
    indice_meilleur = model.cv_results_["rank_test_score"].argmin()
    return round(model.cv_results_["mean_train_score"][indice_meilleur], 3)


def ecart_type_test(model: GridSearchCV) -> np.float64:
    """Retourne l'ecart-type du meilleur modèle."""
    indice_meilleur = model.cv_results_["rank_test_score"].argmin()
    return round(model.cv_results_["std_test_score"][indice_meilleur], 3)


def ecart_type_train(model: GridSearchCV) -> np.float64:
    """Retourne l'ecart-type du meilleur modèle."""
    indice_meilleur = model.cv_results_["rank_test_score"].argmin()
    return round(model.cv_results_["std_train_score"][indice_meilleur], 3)


def parametre(model: GridSearchCV) -> str:
    """Retourne les paramètres du meilleur modèle."""
    indice_meilleur = model.cv_results_["rank_test_score"].argmin()
    return str(model.cv_results_["params"][indice_meilleur])
