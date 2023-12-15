"""
`st_tables` : Le module qui génère les tableaux de l'application 🗃
"""


import streamlit as st
import ast
import polars as pl
from streamlit.delta_generator import DeltaGenerator
from st_plots import *
from sklearn.metrics import r2_score, max_error, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score,f1_score

def write_table(df: pl.DataFrame) -> DeltaGenerator:
    """Retourne une table de données avec des colonnes configurées."""
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


def write_table_ml(chemin_csv) -> DeltaGenerator:
    """Retourne un tableau avec les résultats des modèles"""
    df = pl.read_csv(chemin_csv)
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
                "Score Train 🏋🏻‍♂️",
                min_value=-1,
                max_value=1,
                format="%.2f",
                help="score ∈ [-1,1]",
            ),
            "Ecart-Type Train": "SD Train",
            "Score Test": st.column_config.ProgressColumn(
                "Score Test 👨🏻‍🔬",
                min_value=-1,
                max_value=1,
                format="%.2f",
                help="score ∈ [-1,1]",
            ),
            "Ecart-Type Test": "SD Test",
        },
    )

def clean_param(key):
    """Renomme les valeurs des paramètres"""
    if key == "entrainement__alpha":
        key = "Alpha"
    elif key == "imputation__strategy":
        key = "Stratégie d'imputation"
    elif key == "entrainement__hidden_layer_sizes":
        key = "Hidden layer sizez"
    elif key == "entrainement__max_iter":
        key = "Max iter"
    elif key == "entrainement__solver":
        key = "Solver"
    elif key == "entrainement__C":
        key = "C"
    elif key == "entrainement__n_neighbors":
        key = "N neighbors"
    elif key == "entrainement__max_depth":
        key = "Max depth"
    elif key == "entrainement__n_estimators":
        key = "N estimators"
    elif key == "entrainement__learning_rate":
        key = "Learning rate"
    return key

def parametres(df, place_model) -> DeltaGenerator:
    """Construction du tableau des paramètres"""
    parametres = ast.literal_eval(df["Paramètres"][place_model])
    param = list()
    value = list()
    for key in list(parametres.keys()):
        param.append(clean_param(key))
        value.append(str(parametres[key]))
    tab = pl.DataFrame({"Paramètres ⚒️": param, "Valeur optimale ⭐": value})
    return st.dataframe(tab, hide_index=True)

def write_metrics(type):
    """Metrics principals"""
    if type == "regression":
        df = pl.read_csv("./data/tables/pred_regression.csv")
        expliquee = "unit_price"
    elif type == "classification":
        df = pl.read_csv("./data/tables/pred_classification.csv")
        expliquee = "type"

    models = ["random_forest", "boosting", "ridge", "knn", "mlp", "support_vector"]
    name = ["Random Forest", "Boosting", "Ridge", "K Neighbors", "Réseaux de neurones", "Support Vector"]
    
    metrics_table = {"Modèle 🧰": name}
    
    y_true = df[expliquee]
    if type == "regression":
        metrics_table["Mean Absolute Error ❗"] = [round(mean_absolute_error(y_true, df[model]), 1) for model in models]
        metrics_table["Mean Squared Error ❗❗"] = [round(mean_squared_error(y_true, df[model]), 0) for model in models]
        metrics_table["R2 Score 🔀"] = [round(r2_score(y_true, df[model]), 2) for model in models]
        metrics_table["Max Error 💣"] = [round(max_error(y_true, df[model]), 0) for model in models]
    elif type == "classification":
        metrics_table["Accuracy Score 🏹"] = [round(accuracy_score(y_true, df[model]), 3) for model in models]
        metrics_table["Precision 🔨"] = [round(precision_score(y_true,df[model], average='weighted'), 3) for model in models]
        metrics_table["Recall 🔧"] = [round(recall_score(y_true,df[model], average='weighted'), 3) for model in models]
        metrics_table["F1-Score 🛠️"] = [round(f1_score(y_true,df[model], average='weighted'), 3) for model in models]
    table = pl.DataFrame(metrics_table)
    return st.dataframe(table, hide_index=True)


def write_parameter(chemin_csv, mode):
    """Retourne un tableau avec les paramètres d'un modèle"""
    df = pl.read_csv(chemin_csv)
    df = df.filter(df["Mode"] == mode)
    
    st.subheader("Investigation")
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_model = st.radio(
            "Choisissez un modèle :",
            [
                "Boosting",
                "Random Forest",
                "K Neighbors",
                "Support Vector",
                "Réseaux de neurones",
                "Ridge",
            ],
        )
    with col2:
        if selected_model == "Random Forest":
            parametres(df, 0)
            model = "random_forest"
        elif selected_model == "K Neighbors":
            parametres(df, 1)
            model = "knn"
        elif selected_model == "Réseaux de neurones":
            parametres(df, 2)
            model = "mlp"
        elif selected_model == "Boosting":
            parametres(df, 3)
            model = "boosting"
        elif selected_model == "Ridge":
            parametres(df, 4)
            model = "ridge"
        elif selected_model == "Support Vector":
            parametres(df, 5)
            model = "support_vector"
            
    if mode == "classification":
        display_confusion_matrix(model)
        write_metrics("classification")
    else:
        write_metrics("regression")