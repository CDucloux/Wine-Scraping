"""
`st_tables` : Le module qui génère les tableaux de l'application 🗃
"""


import streamlit as st
import ast
import polars as pl
import pandas as pd
from streamlit.delta_generator import DeltaGenerator
from duckdb import DuckDBPyConnection
from src.modules.app.st_functions import model_mapper_reverse
from sklearn.metrics import (  # type: ignore
    r2_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
)


def write_table(df: pl.DataFrame) -> DeltaGenerator:
    """`write_table`: Retourne une table de données avec des colonnes configurées.

    ---------
    `Parameters`
    --------- ::

        df (pl.DataFrame): # DataFrame mutable

    `Returns`
    --------- ::

        DeltaGenerator

    `Example(s)`
    ---------
    >>> df = load_df()
    >>> write_table(df)
    ... DeltaGenerator()"""
    return st.dataframe(
        data=df,
        hide_index=True,
        column_order=[
            "name",
            "unit_price",
            "picture",
            "capacity",
            "type",
            "millesime",
            "conservation_time",
            "keywords",
            "cepage",
            "bio",
            "is_new",
            "customer_fav",
            "destock",
            "service",
            "avg_temp",
            "alcohol_volume",
            "accords_vins",
            "gouts",
            "oeil",
            "nez",
            "bouche",
            "country",
            "wine_note",
        ],
        column_config={
            "name": "Nom du Vin 🍾",
            "unit_price": st.column_config.NumberColumn(
                "Prix Unitaire 💰",
                help="Le prix du vin à l'unité en euros",
                format="%.2f €",
            ),
            "picture": st.column_config.ImageColumn(
                "Bouteille", help="Prévisualisation de la bouteille", width="medium"
            ),
            "capacity": st.column_config.NumberColumn(
                "Capacité 🚰",
                format="%.3f L",
                help="Capacité de la bouteille (En Litres)",
            ),
            "type": "Type",
            "millesime": st.column_config.NumberColumn("Millésime", format="%d"),
            "conservation_time": st.column_config.NumberColumn(
                "Durée de conservation 📆", format="%d ans"
            ),
            "keywords": st.column_config.ListColumn("Mots-clés"),
            "cepage": "Cépage Majoritaire",
            "bio": st.column_config.CheckboxColumn(
                "Vin Bio 🌱", help="Savoir si le vin possède un label bio"
            ),
            "is_new": st.column_config.CheckboxColumn("Nouveauté 🆕"),
            "customer_fav": st.column_config.CheckboxColumn("Coup de Coeur Client ♥"),
            "destock": st.column_config.CheckboxColumn("Destockage 📦"),
            "service": "Service 🧊",
            "avg_temp": st.column_config.NumberColumn(
                "Température Moyenne",
                help="Température Moyenne de la bouteille",
                format="%.1f degrés",
            ),
            "alcohol_volume": st.column_config.ProgressColumn(
                "Degré d'alcool", min_value=0, max_value=20, format="%.2f°"
            ),
            "accords_vins": "Description 📄",
            "gouts": "Goûts",
            "oeil": "A l'oeil",
            "nez": "Au nez",
            "bouche": "En bouche",
            "country": "Pays d'origine du vin",
            "wine_note": st.column_config.NumberColumn(
                "Note du Vin",
                help="Note du vin /5",
                format="%.1f ⭐",
            ),
        },
    )


def write_table_ml(conn: DuckDBPyConnection, table_name: str) -> DeltaGenerator:
    """`write_table_ml`: Retourne un tableau avec les résultats des modèles.

    ---------
    `Parameters`
    --------- ::

        conn (DuckDBPyConnection): # Connecteur In Memory Database
        table_name (str): # Nom de la table à sélectionner

    `Returns`
    --------- ::

        DeltaGenerator

    `Example(s)`
    ---------
    >>> df = load_df()
    >>> write_table_ml(df)
    ... DeltaGenerator()"""
    df = conn.execute(f"SELECT * FROM {table_name}").pl()
    return st.dataframe(
        data=df,
        hide_index=True,
        column_order=[
            "Modèle",
            "Score Entrainement",
            "Ecart-Type Train",
            "Score Test",
            "Ecart-Type Test",
        ],
        column_config={
            "Modèle": "Modèle 🧰",
            "Score Entrainement": st.column_config.ProgressColumn(
                "Score Train CV 🏋🏻‍♂️",
                min_value=-1,
                max_value=1,
                format="%.2f",
                help="score ∈ [-1,1]",
            ),
            "Ecart-Type Train": "SD Train",
            "Score Test": st.column_config.ProgressColumn(
                "Score Test CV 👨🏻‍🔬",
                min_value=-1,
                max_value=1,
                format="%.2f",
                help="score ∈ [-1,1]",
            ),
            "Ecart-Type Test": "SD Test",
        },
    )


def param_mapper(key: str) -> str:
    """`param_mapper`: Mappe les noms des paramètres 
    optimisés vers des noms plus lisibles.

    ---------
    `Parameters`
    --------- ::

        key (str): # Nom initial des paramètres.

    `Returns`
    --------- ::

        str

    `Example(s)`
    ---------
    >>> df = load_df()
    >>> param_mapper("entrainement__alpha")
    ... 'Alpha'
    ---------
    >>> df = load_df()
    >>> param_mapper("")
    ... 'Le paramètre n'existe pas'"""
    param_mapping = {
        "entrainement__alpha": "Alpha",
        "imputation__strategy": "Stratégie d'imputation",
        "entrainement__hidden_layer_sizes": "Hidden Layer Size",
        "entrainement__max_iter": "Itération maximale",
        "entrainement__solver": "Solveur",
        "entrainement__C": "C",
        "entrainement__n_neighbors": "Nombre de Voisins",
        "entrainement__max_depth": "Profondeur Maximale",
        "entrainement__n_estimators": "N estimators",
        "entrainement__learning_rate": "Learning Rate",
    }
    return param_mapping.get(key, "Le paramètre n'existe pas")


def parametres(df_params: pl.DataFrame, place_model: int) -> DeltaGenerator:
    """`parametres`: Construction du tableau des hyperparamètres 
    optimaux pour chaque modèle.

    ---------
    `Parameters`
    --------- ::

        df_params (pl.DataFrame): 
        # Le DataFrame issu des tables ml_regression et ml_classification
        place_model (int): 
        # Place le modèle à un index particulier

    `Returns`
    --------- ::

        DeltaGenerator

    `Example(s)`
    ---------"""
    parametres = ast.literal_eval(
        df_params.select("Paramètres").to_series()[place_model]
    )
    param = list()
    value = list()
    for key in list(parametres.keys()):
        param.append(param_mapper(key))
        value.append(str(parametres[key]))
    tab = pl.DataFrame({"Paramètres ⚒️": param, "Valeur optimale ⭐": value})
    return st.dataframe(tab, hide_index=True)


def write_metrics(conn: DuckDBPyConnection, type: str) -> DeltaGenerator:
    """`write_metrics`: Metrics principales de Machine Learning.

    ---------
    `Parameters`
    --------- ::

        conn (DuckDBPyConnection): # Connecteur In Memory Database
        type (str): # Regression | Classification

    `Returns`
    --------- ::

        DeltaGenerator

    `Example(s)`
    ---------
    >>> conn = db_connector()
    >>> write_metrics(conn, "regression")
    ... DeltaGenerator()"""
    if type == "regression":
        df = conn.execute("SELECT * FROM pred_regression").pl()
        predicted = "unit_price"
    elif type == "classification":
        df = conn.execute("SELECT * FROM pred_classification").pl()
        predicted = "type"

    models = [
        "random_forest",
        "boosting",
        "ridge",
        "knn",
        "mlp",
        "support_vector",
        "basique",
    ]
    name = [model_mapper_reverse(model) for model in models]
    metrics_table = {"Modèle 🧰": name}

    y_true = df.select(predicted)
    if type == "regression":
        metrics_table["Mean Absolute Error ❗"] = [
            round(mean_absolute_error(y_true, df.select(model)), 1) for model in models
        ]
        metrics_table["Mean Squared Error ❗❗"] = [
            round(mean_squared_error(y_true, df.select(model)), 0) for model in models
        ]
        metrics_table["R2 Score 🔀"] = [
            round(r2_score(y_true, df.select(model)), 2) for model in models
        ]
        metrics_table["Erreur Résiduelle Maximale 💣"] = [
            round(max_error(y_true, df.select(model)), 0) for model in models
        ]
    elif type == "classification":
        metrics_table["Accuracy Score 🏹"] = [
            round(accuracy_score(y_true, df.select(model)), 3) for model in models
        ]
        metrics_table["Precision 🔨"] = [
            round(precision_score(y_true, df.select(model), average="weighted"), 3)
            for model in models
        ]
        metrics_table["Recall 🔧"] = [
            round(recall_score(y_true, df.select(model), average="weighted"), 3)
            for model in models
        ]
        metrics_table["F1-Score 🛠️"] = [
            round(f1_score(y_true, df.select(model), average="weighted"), 3)
            for model in models
        ]
        metrics_table["MCC"] = [
            round(matthews_corrcoef(y_true, df.select(model)), 3) for model in models
        ]
        metrics_table["Rapport 📜"] = [
            classification_report(y_true, df.select(model)) for model in models
        ]
    table = pl.DataFrame(metrics_table)
    if type == "regression":
        table = table.with_columns(
            pl.when(table["Mean Squared Error ❗❗"] > 100000)
            .then(None)
            .otherwise(table["Mean Squared Error ❗❗"])
            .alias("Mean Squared Error ❗❗"),
            pl.when(table["R2 Score 🔀"] < -100)
            .then(None)
            .otherwise(table["R2 Score 🔀"])
            .alias("R2 Score 🔀"),
            pl.when(table["Modèle 🧰"] == "Modèle de base")
            .then(pl.lit("Régression Linéaire"))
            .otherwise(table["Modèle 🧰"])
            .alias(
                "Modèle 🧰",
            ),
        )
    table = table.with_columns(
        pl.when(table["Modèle 🧰"] == "Modèle de base")
        .then(pl.lit("Régression Logistique"))
        .otherwise(table["Modèle 🧰"])
        .alias("Modèle 🧰")
    )
    return st.dataframe(table, hide_index=True)


def write_parameter(conn: DuckDBPyConnection, table_name: str, selected_model: str):
    """`write_parameter`: Retourne un tableau avec les paramètres d'un modèle.

    ---------
    `Parameters`
    --------- ::

        conn (DuckDBPyConnection): # Connecteur In Memory Database
        table_name (str): # Nom de la table à sélectionner
        selected_model (str): # Nom du modèle sélectionné par l'utilisateur

    `Returns`
    --------- ::

        DeltaGenerator

    `Example(s)`
    ---------
    >>> conn = db_connector()
    >>> write_parameter(conn, "ml_regression", "Boosting")
    ... DeltaGenerator()"""
    df_params = conn.execute(f"SELECT * FROM {table_name}").pl()

    if selected_model == "Random Forest":
        params_tbl = parametres(df_params, 0)
    elif selected_model == "K Neighbors":
        params_tbl = parametres(df_params, 1)
    elif selected_model == "Réseaux de neurones":
        params_tbl = parametres(df_params, 2)
    elif selected_model == "Boosting":
        params_tbl = parametres(df_params, 3)
    elif selected_model == "Ridge":
        params_tbl = parametres(df_params, 4)
    elif selected_model == "Support Vector":
        params_tbl = parametres(df_params, 5)
    return params_tbl

def index_best_model_cv(df_cv: pl.DataFrame, df_score : pd.DataFrame) -> int:
    """`index_best_model_cv`: Fonction pour trouver le meilleur 
    modèle selon les score de tests de la CV.

    ---------
    `Parameters`
    --------- ::

        df_cv (pl.DataFrame): # Dataframe des résultats CV
        df_score (pd.DataFrame): # Dataframe résumant les scores

    `Returns`
    --------- ::

        int : # Index du meilleur modèle 

    `Example(s)`
    ---------"""
    best_cv = df_cv["Score Test"].to_list().index(max(df_cv["Score Test"].to_list()))
    best_name = df_cv["Modèle"][best_cv]
    index = df_score["models_name"].to_list().index(best_name)
    return index


def best_model(type: str, conn: DuckDBPyConnection) -> str:
    """`best_model`: Fonction pour conseiller le meilleur modèle.

    - Système de bonus/malus attribué a chaque modèle en fonction
    de plusieurs metrics évaluées.
    - Le modèle ayant le Score Test sur la Cross-Validation gagne
    également des points.

    ---------
    `Parameters`
    --------- ::

        type (str): # Regression ou Classification
        conn (DuckDBPyConnection): # Connecteur In Memory Database

    `Returns`
    --------- ::

        DeltaGenerator

    `Example(s)`
    ---------"""
    models = ["random_forest", "boosting", "ridge", "knn", "mlp", "support_vector"]
    models_name = [
        "Random Forest",
        "Boosting",
        "Ridge",
        "K Neighbors",
        "Réseaux de neurones",
        "Support Vector",
    ]
    df_vide = {"models": models, "models_name": models_name, "score": len(models) * [0]}
    df_score = pd.DataFrame(df_vide)

    if type == "Regression":
        df = conn.execute("SELECT * FROM pred_regression").pl()
        mae = [
            mean_absolute_error(df.select("unit_price"), df.select(model))
            for model in models
        ]
        mse = [
            mean_squared_error(df.select("unit_price"), df.select(model))
            for model in models
        ]
        r2 = [r2_score(df.select("unit_price"), df.select(model)) for model in models]
        me = [max_error(df.select("unit_price"), df.select(model)) for model in models]

        df_score.at[mae.index(min(mae)), "score"] += 1
        df_score.at[mse.index(min(mse)), "score"] += 1
        df_score.at[r2.index(max(r2)), "score"] += 1
        df_score.at[me.index(min(me)), "score"] += 1
        df_score.at[me.index(max(me)), "score"] -= 1

        df2 = conn.execute("SELECT * FROM ml_regression").pl()
        index = index_best_model_cv(df2, df_score)
        df_score.at[index, "score"] += 2

    elif type == "Classification":
        df = conn.execute("SELECT * FROM pred_classification").pl()
        acs = [accuracy_score(df.select("type"), df.select(model)) for model in models]
        prs = [
            precision_score(df.select("type"), df.select(model), average="weighted")
            for model in models
        ]
        res = [
            recall_score(df.select("type"), df.select(model), average="weighted")
            for model in models
        ]
        mac = [
            matthews_corrcoef(df.select("type"), df.select(model)) for model in models
        ]

        df_score.at[acs.index(max(acs)), "score"] += 1
        df_score.at[prs.index(max(prs)), "score"] += 1
        df_score.at[res.index(max(res)), "score"] += 1
        df_score.at[mac.index(max(mac)), "score"] += 1

        df2 = conn.execute("SELECT * FROM ml_classification").pl()
        index = index_best_model_cv(df2, df_score)
        df_score.at[index, "score"] += 2

    best = df_score.at[df_score["score"].idxmax(), "models_name"]
    return best
