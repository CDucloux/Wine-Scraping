"""
Module gérant les fonctions Streamlit de l'application.
"""

import duckdb
import streamlit as st
from annotated_text import annotated_text, annotation
import polars as pl
from pathlib import Path
from src.modules.bear_cleaner import *
from streamlit.delta_generator import DeltaGenerator
from src.modules.ml_models.prediction import *
from PIL import Image
from duckdb import DuckDBPyConnection


@st.cache_resource
def db_connector() -> DuckDBPyConnection:
    """Se connecte à la base de données."""
    root = Path(".").resolve()
    data_folder = root / "data"
    wine_db_path = str(data_folder / "DB" / "models_db.db")
    connection = duckdb.connect(wine_db_path)
    return connection


@st.cache_data
def load_df() -> pl.DataFrame:
    """Charge notre DataFrame clean."""
    root = Path(".").resolve()
    data_folder = root / "data"
    df = pl.read_json(data_folder / "vins.json")
    df = super_pipe(df)
    return df


# TODO: expliciter les variables d'input pour mypy pour la fonction load_main_df


@st.cache_data
def load_main_df(
    _df: pl.DataFrame,
    selected_wines: list[str],
    prices,
    filter_bio,
    filter_new,
    filter_fav,
    user_input,
) -> pl.DataFrame:
    """Charge le dataframe filtré"""
    main_df = (
        _df.filter(pl.col("type").is_in(selected_wines))
        .filter(pl.col("unit_price") > prices[0])
        .filter(pl.col("unit_price") < prices[1])
        .filter(pl.col("bio").is_in(filter_bio))
        .filter(pl.col("is_new").is_in(filter_new))
        .filter(pl.col("customer_fav").is_in(filter_fav))
        .filter(pl.col("name").str.contains(user_input))
    )
    return main_df


def page_config() -> None:
    """Configure le titre et le favicon de l'application."""
    return st.set_page_config(page_title="Wine Scraper", page_icon="🍇")


def remove_white_space() -> DeltaGenerator:
    """Utilise du CSS pour retirer de l'espace non-utilisé"""
    return st.markdown(
        """
        <style>
                .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 0rem;
                    padding-right: 0rem;
                }
                .st-emotion-cache-16txtl3{
                    padding-top: 0.5rem;
                    padding-right: 1rem;
                    padding-bottom: 1rem;
                    padding-left: 1rem;
                }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_wine_selector() -> list[str]:
    """Permet de sélectionner un type de vin."""
    wine_types = ["Vin Blanc", "Vin Rouge", "Vin Rosé"]
    return st.multiselect(
        "Sélectionnez un type de vin :",
        wine_types,
        default="Vin Rouge",
        placeholder="Choisir un type de vin",
    )


def sidebar_prices_slider(df: pl.DataFrame):
    """Permet de choisir un intervalle de prix."""
    return st.slider(
        "Sélectionnez un intervalle de prix :",
        df.select("unit_price").min().item(),
        df.select("unit_price").max().item(),
        (5.0, 400.0),
        format="%.2f€",
    )


def sidebar_checkbox_bio():
    """Une case à cocher pour n'inclure que les vins bios."""
    bio = st.sidebar.checkbox("N'inclure que les vins Bio 🌿")
    if bio:
        filter_bio = 1
    else:
        filter_bio = {0, 1}
    return filter_bio


def sidebar_checkbox_new():
    """Une case à cocher pour n'inclure que les nouveautés."""
    new = st.sidebar.checkbox("N'inclure que les nouveautés 🆕")
    if new:
        filter_new = 1
    else:
        filter_new = {0, 1}
    return filter_new


def sidebar_checkbox_fav():
    """Une case à cocher pour n'inclure que les coups de coeur client."""
    fav = st.sidebar.checkbox("N'inclure que les Coups de Coeur")
    if fav:
        filter_fav = 1
    else:
        filter_fav = {0, 1}
    return filter_fav


def sidebar_input_wine() -> str:
    """Un user input permettant de rechercher un nom de vin."""
    user_input = st.text_input("Recherche par nom de vin :").upper()
    return user_input


def main_wine_metric(df: pl.DataFrame, wine_type: str) -> DeltaGenerator:
    """Permet d'obtenir une métrique du nombre de vins selon le type de vin."""
    wine_count = (
        df.group_by(pl.col("type"))
        .count()
        .sort("count", descending=True)
        .filter(pl.col("type") == wine_type)
        .select("count")
        .item()
    )
    news = (
        df.filter(pl.col("is_new") == 1)
        .group_by("type")
        .count()
        .sort("count", descending=True)
        .filter(pl.col("type") == wine_type)
        .select("count")
        .item()
    )

    if wine_type == "Vin Rouge":
        colored_text = f":red[{wine_type}]"
    elif wine_type == "Vin Blanc":
        colored_text = f":orange[{wine_type}]"
    else:
        colored_text = wine_type
    return st.metric(
        colored_text,
        wine_count,
        f"{news} nouveautés !",
    )


def write_price(df: pl.DataFrame, selected_wines: list[str]) -> None:
    """Retourne le prix moyen d'un vin de la sélection."""
    mean_price = df.select(pl.col("unit_price")).mean().item()
    if mean_price == None:
        return st.write("Le prix moyen d'un vin de la sélection est *incalculable*.")
    else:
        return st.write(
            "Le prix moyen d'un",
            " / ".join(selected_wines).lower(),
            " de la sélection est de ",
            f'`{str(round(mean_price, 2)).replace(".", ",")} €`.',
        )


def scale_selector():
    """Crée un radio button pour sélectionner l'échelle."""
    return st.radio(
        "Sélectionner une *échelle*",
        ["$y$", "$\log(y)$"],
    )


# TODO: corriger le color_selector qui merde.


def color_selector(selected_wines: list[str]) -> list[str]:
    """Permet de choisir une couleur selon le vin sélectionné pour le scatter plot des vins."""
    red, white, pink = "#ff4b4b", "#f3b442", "#ff8fa3"
    if selected_wines == ["Vin Rouge"]:
        colors = [red]
    elif selected_wines == ["Vin Blanc"]:
        colors = [white]
    elif selected_wines == ["Vin Rosé"]:
        colors = [pink]
    elif selected_wines == ["Vin Rouge", "Vin Blanc"]:
        colors = [red, white]
    elif selected_wines == ["Vin Rouge", "Vin Rosé"]:
        colors = [red, pink]
    elif selected_wines == ["Vin Blanc", "Vin Rouge"]:
        colors = [red, white]
    elif selected_wines == ["Vin Blanc", "Vin Rosé"]:
        colors = [white, pink]
    elif selected_wines == ["Vin Rosé", "Vin Rouge"]:
        colors = [red, pink]
    elif selected_wines == ["Vin Rosé", "Vin Blanc"]:
        colors = [white, pink]
    elif selected_wines == ["Vin Rouge", "Vin Blanc", "Vin Rosé"]:
        colors = [red, white, pink]
    else:
        colors = [red, white, pink]
    return colors


def custom_radio_css() -> None:
    """Repositionne les boutons radio (colonne --> ligne)."""
    return st.write(
        "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
        unsafe_allow_html=True,
    )


def info() -> DeltaGenerator:
    return st.info(
        "L'ensemble de cet onglet est statique, la barre de paramètres n'influera pas sur les données.",
        icon="ℹ️",
    )


def authors() -> DeltaGenerator:
    """Crée la page 6 qui inclue nos noms 😎."""
    st.balloons()
    with st.expander("Découvrir les `auteurs` de l'application"):
        st.markdown(
            """
- *Corentin DUCLOUX* : https://github.com/CDucloux 
- *Guillaume DEVANT* : https://github.com/devgui37
"""
        )
    # TODO: utiliser pathlib ici, pas de string relatif.
    image = Image.open("./img/img_vins.jpg")
    st.image(image)
    return DeltaGenerator


def model_mapper(model_name: str) -> str:
    """Mappe le nom des modèles à ceux contenus dans la base de données."""
    model_names_mapping = {
        "Random Forest": "random_forest",
        "Boosting": "boosting",
        "Ridge": "ridge",
        "Réseaux de neurones": "mlp",
        "K Neighbors": "knn",
        "Support Vector": "support_vector",
    }
    return model_names_mapping.get(model_name, "Le modèle n'existe pas")


def model_selector() -> str:
    """Permet de sélectionner un modèle de Machine learning."""
    return st.selectbox(
        "Modèle :",
        (
            "Random Forest",
            "Boosting",
            "Ridge",
            "Réseaux de neurones",
            "K Neighbors",
            "Support Vector",
        ),
    )


def format_prediction(prediction: float | str, truth: float | str) -> str:
    """Formate le résultat brut de la prédiction dans l'application (soit le prix, soit le type de vin)."""
    if type(prediction) == float and type(truth) == float:
        if (prediction / truth) > 0.8 and (prediction / truth) < 1.2:
            format_prediction = f"✅ {round(prediction,2)} €".replace(".", ",")
        else:
            format_prediction = f"❌ {round(prediction,2)} €".replace(".", ",")
    else:
        if truth == prediction:
            format_prediction = f"✅ {prediction}"
        else:
            format_prediction = f"❌ {prediction}"
    return format_prediction


def popover_prediction(
    prediction: float, truth: float
) -> tuple[DeltaGenerator, DeltaGenerator]:
    """Renvoie un message d'avertissement selon que le prix prédit soit supérieur ou inférieur au prix réel."""
    if prediction - truth < 0:
        text = f"🚨 Le prix prédit est {abs(round(prediction-truth,2))} € **inférieur** au prix réel !"
    elif prediction - truth > 0:
        text = f"🚨 Le prix prédit est {abs(round(prediction-truth,2))} € **supérieur** au prix réel !"
    return st.error(text.replace(".", ",")), st.caption(
        "$^*$ Il est possible que le prix prédit soit **très loin de la réalité**, voire même **négatif**, en dépit de nos efforts."
    )


def get_names(conn: DuckDBPyConnection) -> list[str]:
    """Récupère les noms des vins qui ont été prédits par le modèle."""
    result = conn.sql("SELECT name FROM pred_regression")
    names = [row[0] for row in result.fetchall()]
    return names


def get_value(conn: DuckDBPyConnection, column: str, table_name: str, wine_name: str):
    """Récupère la colonne d'une table filtrée selon le nom d'un vin, c'est à dire une valeur.

    La colonne peut être :

    - unit_price
    - type
    - un des 6 modèles de Machine Learning

    """
    return conn.sql(
        f"SELECT {column} FROM {table_name} WHERE name = '{wine_name}'"
    ).fetchone()[0]
