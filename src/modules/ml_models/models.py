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


def data_model(path: str, target: str) -> pl.DataFrame:
    """`data_model`: Importe le JSON, le transforme en dataframe, le nettoie et le prépare pour le ML.

    ---------
    `Parameters`
    --------- ::

        path (str): # Chemin vers les données
        target (str): # Variable à prédire

    `Returns`
    --------- ::

        pl.DataFrame

    `Example(s)`
    ---------

    >>> data_model(path="./data/vins.json", target="type")
    ... shape: (4_006, 40)"""
    df_brut = pl.read_json(path)
    df = super_pipe(df_brut)
    df = df.filter(pl.col(target).is_not_null())
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
    ---------"""
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

    >>> model_rf(x_train=X_train, y_train=y_train, mode = "regression")
    ... Entrainement du modèle : Random Forest
    ... GridSearchCV(estimator=Pipeline(steps=[('imputation', SimpleImputer()),
    ...                                   ('echelle', MinMaxScaler()),
    ...                                   ('entrainement',
    ...                                    RandomForestRegressor())]),
    ...         n_jobs=-1,
    ...         param_grid={'entrainement__max_depth': range(1, 10),
    ...                     'entrainement__n_estimators': range(10, 50, 10),
    ...                     'imputation__strategy': ['mean', 'median',
    ...                                              'most_frequent']},
    ...         return_train_score=True)
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

    >>> model_knn(x_train=X_train, y_train=y_train, mode = "classification")
    ... Entrainement du modèle : K Neighbors
    ... GridSearchCV(estimator=Pipeline(steps=[('imputation', SimpleImputer()),
    ...                                ('echelle', MinMaxScaler()),
    ...                                ('entrainement',
    ...                                 KNeighborsClassifier())]),
    ...      n_jobs=-1,
    ...      param_grid={'entrainement__n_neighbors': range(2, 15),
    ...                  'imputation__strategy': ['mean', 'median',
    ...                                           'most_frequent']},
    ...      return_train_score=True)"""
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
    
    >>> model_boost(x_train=X_train, y_train=y_train, mode = "classification")
    ... Entrainement du modèle : Boosting
    ... GridSearchCV(estimator=Pipeline(steps=[('imputation', SimpleImputer()),
    ...                                    ('echelle', MinMaxScaler()),
    ...                                    ('entrainement',
    ...                                     GradientBoostingClassifier())]),
    ...          n_jobs=-1,
    ...          param_grid={'entrainement__learning_rate': (0.005, 0.01, 0.1, 0.5),
    ...                      'entrainement__n_estimators': (50, 100, 150, 200, 400),
    ...                      'imputation__strategy': ['mean', 'median',
    ...                                               'most_frequent']},
    ...          return_train_score=True)
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

    >>> model_mlp(x_train=X_train, y_train=y_train, mode = "classification")
    ... Entrainement du modèle : Réseaux de neurones
    ... GridSearchCV(estimator=Pipeline(steps=[('imputation', SimpleImputer()),
    ...                                    ('echelle', MinMaxScaler()),
    ...                                    ('entrainement', MLPClassifier())]),
    ...          n_jobs=-1,
    ...          param_grid={'entrainement__hidden_layer_sizes': [(100,), (50, 50),
    ...                                                           (60, 60)],
    ...                      'entrainement__max_iter': [1000],
    ...                      'entrainement__solver': ['adam'],
    ...                      'imputation__strategy': ['mean', 'median',
    ...                                               'most_frequent']},
    ...          return_train_score=True)"""
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

    >>> model_ridge(x_train=X_train, y_train=y_train, mode = "classification")
    ... Entrainement du modèle : Ridge
    ... GridSearchCV(estimator=Pipeline(steps=[('imputation', SimpleImputer()),
    ...                                    ('echelle', MinMaxScaler()),
    ...                                    ('entrainement', RidgeClassifier())]),
    ...          n_jobs=-1,
    ...          param_grid={'entrainement__alpha': [0.015625, 0.03125, 0.0625,
    ...                                              0.125, 0.25, 0.5, 1, 2, 4, 8,
    ...                                              16, 32, 64],
    ...                      'imputation__strategy': ['mean', 'median',
    ...                                               'most_frequent']},
    ...          return_train_score=True)"""
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

    >>> model_svm(x_train=X_train, y_train=y_train, mode = "classification")
    ... Entrainement du modèle : Support Vector
    ... GridSearchCV(estimator=Pipeline(steps=[('imputation', SimpleImputer()),
    ...                                    ('echelle', MinMaxScaler()),
    ...                                    ('entrainement', SVC())]),
    ...          n_jobs=-1,
    ...          param_grid={'entrainement__C': array([6.25000000e-02, 1.68237524e-01, 4.52861832e-01, 1.21901365e+00,
    ...    3.28134142e+00, 8.83271611e+00, 2.37759086e+01, 6.40000000e+01,
    ...    1.72275225e+02, 4.63730516e+02, 1.24826998e+03, 3.36009362e+03,
    ...    9.04470130e+03, 2.43465304e+04, 6.55360000e+04]),
    ...                      'imputation__strategy': ['mean', 'median',
    ...                                               'most_frequent']},
    ...          return_train_score=True)"""
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

    >>> train_model(x_train=X_train, y_train=y_train, mode = "classification")
    ... Entrainement du modèle : K Neighbors
    ... Entrainement du modèle : Random Forest
    ... Entrainement du modèle : Boosting
    ... Entrainement du modèle : Ridge
    ... Entrainement du modèle : Support Vector
    ... Entrainement du modèle : Réseaux de neurones
    ... {'model_knn': GridSearchCV(estimator=Pipeline(steps=[('imputation', SimpleImputer()),
    ...                                     ('echelle', MinMaxScaler()),
    ...                                     ('entrainement',
    ...                                      KNeighborsClassifier())]),
    ...           n_jobs=-1,
    ...           param_grid={'entrainement__n_neighbors': range(2, 15),
    ...                       'imputation__strategy': ['mean', 'median',
    ...                                                'most_frequent']},
    ... 'model_rf' : ...}"""
    return {
        "model_knn": model_knn(x_train, y_train, mode),
        "model_rf": model_rf(x_train, y_train, mode),
        "model_boost": model_boost(x_train, y_train, mode),
        "model_ridge": model_ridge(x_train, y_train, mode),
        "model_svm": model_svm(x_train, y_train, mode),
        "model_mlp": model_mlp(x_train, y_train, mode),
    }


def score_test(model: GridSearchCV) -> np.float64:
    """`score_test`: Retourne le score de test du meilleur modèle.
    Note : le meilleur modèle est celui ayant le score de test minimal
    ---------
    `Parameters`
    --------- ::

        model (GridSearchCV): # Modèle

    `Returns`
    --------- ::

        np.float64

    `Example(s)`
    ---------

    >>> score_test(model["model_rf"])
    ... 0.69"""
    indice_meilleur = model.cv_results_["rank_test_score"].argmin()
    return round(model.cv_results_["mean_test_score"][indice_meilleur], 3)


def score_entrainement(model: GridSearchCV) -> np.float64:
    """`score_entrainement`: Retourne le score d'entrainement du meilleur modèle.
    Note : le meilleur modèle est celui ayant le score de test minimal
    
    ---------
    `Parameters`
    --------- ::

        model (GridSearchCV): # Modèle

    `Returns`
    --------- ::

        np.float64

    `Example(s)`
    ---------

    >>> score_entrainement(model["model_rf"])
    ... 0.69"""
    indice_meilleur = model.cv_results_["rank_test_score"].argmin()
    return round(model.cv_results_["mean_train_score"][indice_meilleur], 3)


def ecart_type_test(model: GridSearchCV) -> np.float64:
    """`ecart_type_test`: Retourne l'ecart-type de test du meilleur modèle.
    Note : le meilleur modèle est celui ayant le score de test minimal
    
    ---------
    `Parameters`
    --------- ::

        model (GridSearchCV): # Modèle

    `Returns`
    --------- ::

        np.float64

    `Example(s)`
    ---------

    >>> ecart_type_test(model["model_rf"])
    ... 0.007"""
    indice_meilleur = model.cv_results_["rank_test_score"].argmin()
    return round(model.cv_results_["std_test_score"][indice_meilleur], 3)


def ecart_type_train(model: GridSearchCV) -> np.float64:
    """`ecart_type_train`: Retourne l'ecart-type d'entrainement du meilleur modèle.
    Note : le meilleur modèle est celui ayant le score de test minimal
    
    ---------
    `Parameters`
    --------- ::

        model (GridSearchCV): # Modèle

    `Returns`
    --------- ::

        np.float64

    `Example(s)`
    ---------

    >>> ecart_type_train(model["model_rf"])
    ... 0.007"""
    indice_meilleur = model.cv_results_["rank_test_score"].argmin()
    return round(model.cv_results_["std_train_score"][indice_meilleur], 3)


def parametre(model: GridSearchCV) -> str:
    """`parametre`: Retourne les paramètres du meilleur modèle.
    Note : le meilleur modèle est celui ayant le score de test minimal
    
    ---------
    `Parameters`
    --------- ::

        model (GridSearchCV): # Modèle

    `Returns`
    --------- ::

        str

    `Example(s)`
    ---------

    >>> parametre(model["model_rf"])
    ... '{'entrainement__max_depth': 9, 'entrainement__n_estimators': 30, 'imputation__strategy': 'median'}'"""
    indice_meilleur = model.cv_results_["rank_test_score"].argmin()
    return str(model.cv_results_["params"][indice_meilleur])
