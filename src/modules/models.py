"""
Module compilant les modèles
"""

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from prettytable import PrettyTable
import numpy as np
import polars as pl
from src.modules.bear_cleaner import super_pipe

#Preparation
def prep_str(df):
    """Transforme les variables texte en nombre pour effectuer le ML"""
    df = df.to_pandas()
    df_prep = df
    le = LabelEncoder()
    df_prep["keyword_1"] = le.fit_transform(df["keyword_1"])
    df_prep["keyword_2"] = le.fit_transform(df["keyword_2"])
    df_prep["keyword_3"] = le.fit_transform(df["keyword_3"])
    df_prep["cepage"] = le.fit_transform(df["cepage"])
    df_prep["par_gouts"] = le.fit_transform(df["par_gouts"])
    df_prep["service"] = le.fit_transform(df["service"])
    df_prep["type"] = le.fit_transform(df["type"])
    df_prep["country"] = le.fit_transform(df["country"])
    return df_prep

def data_model(chemin:str):
    """Import le json, le transforme en dataframe, le nettoie et le prépare pour le ML"""
    df = pl.read_json(chemin)
    df = super_pipe(df)
    df = df.filter(pl.col("unit_price").is_not_null())
    df = prep_str(df)
    return df

#Modèles
def model_rf(x_train, y_train):
    """
    paramètres optimisés :
    - n_estimators
    - max_depth
    """
    model = Pipeline(
        [
            ("imputation", SimpleImputer()),
            ("echelle", MinMaxScaler()),
            ("entrainement", RandomForestRegressor()),
        ]
    )
    cv = GridSearchCV(
        estimator=model,
        param_grid={
            "imputation__strategy": ["mean", "median", "most_frequent"],
            "entrainement__n_estimators": range(10, 50, 10),
            "entrainement__max_depth": range(1, 10, 1)
        },
        n_jobs=-1,
    )
    cv.fit(x_train, y_train)
    return cv


def model_knn(x_train, y_train):
    """
    paramètres optimisés :
    -n_neighbors
    """
    model = Pipeline(
        [
            ("imputation", SimpleImputer()),
            ("echelle", MinMaxScaler()),
            ("entrainement", KNeighborsRegressor()),
        ]
    )
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


def model_boost(x_train, y_train):
    """
    paramètres optimisés :
    -learning_rate
    -n_estimators
    """
    model = Pipeline(
        [
            ("imputation", SimpleImputer()),
            ("echelle", MinMaxScaler()),
            ("entrainement", GradientBoostingRegressor()),
        ]
    )
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


def model_ridge(x_train, y_train):
    """
    paramètres optimisés :
    -alpha
    
    /!\ La régression Ridge ajoute une pénalité (régularisation) à la 
    régression linéaire standard en modifiant la fonction objectif.
    """
    model = Pipeline(
        [
            ("imputation", SimpleImputer()),
            ("echelle", MinMaxScaler()),
            ("entrainement", Ridge()),
        ]
    )
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


def model_svm(x_train, y_train):
    """
    paramètres optimisés :
    - C
    - epsilon
    """
    model = Pipeline(
        [
            ("imputation", SimpleImputer()),
            ("echelle", MinMaxScaler()),
            ("entrainement", SVR()),
        ]
    )
    cv = GridSearchCV(
        estimator=model,
        param_grid={
            "imputation__strategy": ["mean", "median", "most_frequent"],
            "entrainement__C": np.logspace(-4, 16, 15, base=2),
            "entrainement__epsilon": [0.1, 0.4, 0.7],
        },
        n_jobs=-1,
    )
    cv.fit(x_train, y_train)
    return cv


def model_mlp(x_train, y_train):
    """
    paramètres optimisés :
    -hidden_layer_sizes
    -max_iter
    
    A CONSULTER : https://playground.tensorflow.org/#activation=relu&batchSize=1&dataset=spiral&regDataset=reg-plane&learningRate=0.01&regularizationRate=0&noise=0&networkShape=4,1&seed=0.70523&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false
    """
    model = Pipeline(
        [
            ("imputation", SimpleImputer()),
            ("echelle", MinMaxScaler()),
            ("entrainement", MLPRegressor()),
        ]
    )
    cv = GridSearchCV(
        estimator=model,
        param_grid={
            "imputation__strategy": ["mean", "median", "most_frequent"],
            "entrainement__hidden_layer_sizes": [(100,), (50,50), (60,60)],
            "entrainement__solver": ["adam"],
            "entrainement__max_iter": [1000],
        },
        n_jobs=-1,
    )
    cv.fit(x_train, y_train)
    return cv


def train_model(x_train, y_train):
    """Fonction entrainant tous les modèles"""
    return {
        "model_knn": model_knn(x_train, y_train),
        "model_rf": model_rf(x_train, y_train),
        "model_boost": model_boost(x_train, y_train),
        "model_ridge": model_ridge(x_train, y_train),
        "model_svm": model_svm(x_train, y_train),
        "model_mlp": model_mlp(x_train, y_train),
    }

#Résultats
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