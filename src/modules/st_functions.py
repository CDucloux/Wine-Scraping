"""
Module gérant les fonctions Streamlit de l'application.
"""
import streamlit as st
import polars as pl
from pathlib import Path
from bear_cleaner import *
from streamlit.delta_generator import DeltaGenerator

# TODO: Faire une fonction load_data avec le décorateur @st.cache_data pour éviter de recharger le df tt le temps
# TODO: -- Impossible actuellement...Regarder comment changer la couleur dans un DataFrame, notamment pour prix & Type de vin
# TODO: Faire une carte de la provenance des vins


@st.cache_data
def load_df() -> pl.DataFrame:
    """Charge notre DataFrame clean."""
    root = Path(".").resolve()
    data_folder = root / "data"
    df = pl.read_json(data_folder / "vins.json")
    df = super_pipe(df)
    return df


def page_config() -> None:
    """Configure le titre et le favicon de l'application."""
    return st.set_page_config(page_title="Vins à la carte", page_icon="🍇")


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
                .css-1544g2n.eczjsme4 {
                    padding-top: 0.75rem;
                    padding-right: 1rem;
                    padding-bottom: 1rem;
                    padding-left: 1rem;
                }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_wine_selector():
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
        (10.0, 400.0),
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


def sidebar_year_selector(df: pl.DataFrame) -> list:
    """Un multisélecteur permettant de choisir entre différents millésimes."""
    return st.multiselect(
        "Millésime",
        df.select(pl.col("millesime").unique()).to_series().to_list(),
        placeholder="Choisissez une/plusieurs années",
        default=[2019, 2020, 2021, 2022],
    )


def main_wine_metric(df: pl.DataFrame, wine_type: str) -> list:
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
