"""
`st_tables` : Le module qui génère les tableaux de l'application 🗃
"""


import streamlit as st
import ast
import polars as pl
from streamlit.delta_generator import DeltaGenerator


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


def write_table_ml(chemin_csv, mode) -> DeltaGenerator:
    """Retourne un tableau avec les résultats des modèles"""
    df = pl.read_csv(chemin_csv)
    df = df.filter(df["Mode"] == mode)
    return st.dataframe(
        data=df,
        hide_index=True,
        column_order=[
            "Modèle",
            "Score Entrainement",
            "Ecart-Type Train",
            "Score Test",
            "Ecart-Type Test",
            "Score Test data",
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
            "Score Test data": "Métrique 🏭",
        },
    )


def parametres(df, place_model):
    """Construction du tableau des paramètres"""
    parametres = ast.literal_eval(df["Paramètres"][place_model])
    param = list()
    value = list()
    for key in list(parametres.keys()):
        param.append(key)
        value.append(str(parametres[key]))
    tab = pl.DataFrame({"Paramètres ⚒️": param, "Valeur optimale ⭐": value})
    return st.dataframe(tab, hide_index=True)


def write_parameter(chemin_csv, mode):
    """Retourne un tableau avec les paramètres d'un modèle"""
    df = pl.read_csv(chemin_csv)
    df = df.filter(df["Mode"] == mode)

    col1, col2 = st.columns([1, 2])
    with col1:
        selected_model = st.radio(
            "Consultez les paramètres optimaux",
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
        elif selected_model == "K Neighbors":
            parametres(df, 1)
        elif selected_model == "Réseaux de neurones":
            parametres(df, 2)
        elif selected_model == "Boosting":
            parametres(df, 3)
        elif selected_model == "Ridge":
            parametres(df, 4)
        elif selected_model == "Support Vector":
            parametres(df, 5)
