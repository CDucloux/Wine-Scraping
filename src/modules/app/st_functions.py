"""
Module gérant les fonctions Streamlit de l'application.
"""

import duckdb
import streamlit as st
import polars as pl
from pathlib import Path
from PIL import Image
from streamlit.delta_generator import DeltaGenerator
from duckdb import DuckDBPyConnection
from src.modules.bear_cleaner import *  # type: ignore


@st.cache_resource
def db_connector() -> DuckDBPyConnection:
    """Connecteur à la base de données."""
    connection = duckdb.connect(database=":memory:")
    return connection


def load_tables(connection: DuckDBPyConnection) -> None:
    """Charge l'ensemble des tables en csv dans la base de données In-memory."""
    root = Path(".").resolve()
    data_folder = root / "data"
    tables_folder = data_folder / "tables"
    pred_reg = str(tables_folder / "pred_regression.csv")
    pred_class = str(tables_folder / "pred_classification.csv")
    ml_reg = str(tables_folder / "result_ml_regression.csv")
    ml_class = str(tables_folder / "result_ml_classification.csv")
    """Crée la table des prédictions pour la régression."""
    connection.execute(
        """
    CREATE OR REPLACE TABLE pred_regression AS
        SELECT * FROM read_csv_auto(?, header = true);
    """,
        [pred_reg],
    )
    """Crée la table des prédictions pour la classification."""
    connection.execute(
        """
    CREATE OR REPLACE TABLE pred_classification AS
        SELECT * FROM read_csv_auto(?, header = true);
    """,
        [pred_class],
    )
    """Crée la table des résultats pour la classification."""
    connection.execute(
        """
    CREATE OR REPLACE TABLE ml_regression AS
        SELECT * FROM read_csv_auto(?, header = true);
    """,
        [ml_reg],
    )
    """Crée la table des résultats pour la classification."""
    connection.execute(
        """
    CREATE OR REPLACE TABLE ml_classification AS
        SELECT * FROM read_csv_auto(?, header = true);
    """,
        [ml_class],
    )
    return None


@st.cache_data
def load_df() -> pl.DataFrame:
    """`load_df`: Charge notre DataFrame clean statique.

    `Returns`
    --------- ::

        pl.DataFrame

    `Example(s)`
    ---------

    >>> load_df()
    ... #_test_return_"""
    root = Path(".").resolve()
    data_folder = root / "data"
    df = pl.read_json(data_folder / "vins.json")
    df = super_pipe(df)
    return df


@st.cache_data
def load_main_df(
    _df: pl.DataFrame,
    selected_wines: list[str],
    prices: tuple[float, float],
    filter_bio: set[int],
    filter_new: set[int],
    filter_fav: set[int],
    user_input: str,
) -> pl.DataFrame:
    """`load_main_df`: Charge le DataFrame clean, mais mutable avec possibilité de filtre.

    ---------
    `Parameters`
    --------- ::

        _df (pl.DataFrame): # Le DataFrame clean
        selected_wines (list[str]): # Type(s) de vin(s) sélectionné(s)
        prices (tuple[float, float]): # Le prix min et max sélectionné
        filter_bio (set[int]): # Filtre sur les vins bios
        filter_new (set[int]): # Filtre sur les nouveautés
        filter_fav (set[int]): # Filtre sur les vins favoris
        user_input (str): # Recherche de vin spécifique

    `Returns`
    --------- ::

        pl.DataFrame

    `Example(s)`
    ---------

    >>> load_main_df()
    ... #_test_return_"""
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


def custom_radio_css() -> None:
    """Repositionne les boutons radio (colonne vers ligne)."""
    return st.write(
        "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
        unsafe_allow_html=True,
    )


def main_wine_metric(df: pl.DataFrame, wine_type: str) -> DeltaGenerator:
    """`main_wine_metric`: Permet d'obtenir une métrique du nombre de vins et du nombre de nouveautés associées selon le type de vin.

    ---------
    `Parameters`
    --------- ::

        df (pl.DataFrame): # DataFrame statique
        wine_type (str): # Type de vin

    `Returns`
    --------- ::

        DeltaGenerator

    `Example(s)`
    ---------

    >>> main_wine_metric()
    ... #_test_return_"""
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
    """Retourne le prix moyen d'un vin de la sélection ou indique l'impossibilité de le calculer."""
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


def info() -> DeltaGenerator:
    """Retourne des informations sur une page."""
    return st.info(
        "L'ensemble de cet onglet est statique, la barre de paramètres n'influera pas sur les données.",
        icon="ℹ️",
    )


def authors() -> tuple[DeltaGenerator, DeltaGenerator, DeltaGenerator]:
    """Crée la page 6 qui inclue nos noms 😎."""
    image = Image.open("./img/img_vins.jpg")
    return (
        st.balloons(),
        st.expander("Découvrir les `auteurs` de l'application").markdown(
            """
- *Corentin DUCLOUX* : https://github.com/CDucloux 
- *Guillaume DEVANT* : https://github.com/devgui37
"""
        ),
        st.image(image),
    )


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


def model_mapper_reverse(model_name: str) -> str:
    """Mappe les noms de modèles de la base de données à ceux "réels"."""
    model_names_mapping = {
        "random_forest": "Random Forest",
        "boosting": "Boosting",
        "ridge": "Ridge",
        "mlp": "Réseaux de neurones",
        "knn": "K Neighbors",
        "support_vector": "Support Vector",
    }
    return model_names_mapping.get(model_name, "Le modèle n'existe pas")


# TODO: changer le 0.8 et 1.2 en tant qu'Enum


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
        if (prediction / truth) > 0.8 and (prediction / truth) < 1.2:
            text = f"✔ Le prix prédit est {abs(round(prediction-truth,2))} € **inférieur** au prix réel, soit une différence acceptable !"
        else:
            text = f"🚨 Le prix prédit est {abs(round(prediction-truth,2))} € **inférieur** au prix réel, soit une importante différence !"
    elif prediction - truth > 0:
        if (prediction / truth) > 0.8 and (prediction / truth) < 1.2:
            text = f"✔ Le prix prédit est {abs(round(prediction-truth,2))} € **supérieur** au prix réel, soit une différence acceptable !"
        else:
            text = f"🚨 Le prix prédit est {abs(round(prediction-truth,2))} € **supérieur** au prix réel, soit une importante différence !"
    else:
        text = "Le prix prédit est strictement égal au prix réel !"
    return st.error(text.replace(".", ",")), st.caption(
        "$^*$ Il est possible que le prix prédit soit **très loin de la réalité**, voire même **négatif**, en dépit de nos efforts."
    )


def get_names(conn: DuckDBPyConnection) -> list[str]:
    """Récupère les noms des vins qui ont été prédits par le modèle."""
    result = conn.execute("SELECT name FROM pred_regression")
    names = [row[0] for row in result.fetchall()]
    return names


def get_value(
    conn: DuckDBPyConnection, column: str, table_name: str, wine_name: str
) -> float | str:
    """Récupère la colonne d'une table filtrée selon le nom d'un vin, c'est à dire une valeur.

    La colonne peut être :

    - unit_price
    - type
    - un des 6 modèles de Machine Learning

    """
    query = conn.execute(
        f"SELECT {column} FROM {table_name} WHERE name = ?", [wine_name]
    )
    value = query.fetchall()[0][0]
    return value
