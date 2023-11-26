"""
Module gérant les fonctions Streamlit de l'application.
"""
import ast
import duckdb
import streamlit as st
from annotated_text import annotated_text, annotation
import polars as pl
import plotly.express as px
import plotly.figure_factory as ff
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
    wine_db_path = str(data_folder / "DB" / "dt.db")
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


# TODO: expliciter les variables d'input pour mypy.


@st.cache_data
def load_main_df(
    _df: pl.DataFrame,
    selected_wines: list[str],
    prices,
    filter_bio,
    filter_new,
    filter_fav,
    user_input,
    years: list[int],
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
        .filter(pl.col("millesime").is_in(years))
    )
    return main_df


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


def color_selector(selected_wines: list[str]) -> list[str]:
    """Permet de choisir une couleur selon le vin sélectionné."""
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


def scale_selector():
    """Crée un radio button pour sélectionner l'échelle."""
    return st.radio(
        "Sélectionner une *échelle*",
        ["$y$", "$\log(y)$"],
    )


def custom_radio_css() -> None:
    """Repositionne les boutons radio (colonne --> ligne)."""
    return st.write(
        "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
        unsafe_allow_html=True,
    )


def warnings(df: pl.DataFrame, selected_wines: list[str]) -> DeltaGenerator | None:
    """Renvoie des messages d'avertissement spécifiques quand le dataframe modifié à cause de la sidebar ne génère pas de données."""
    if not selected_wines:
        return st.warning(
            "Attention, aucun type de vin n'a été selectionné !", icon="🚨"
        )
    elif len(df) == 0:
        return st.warning(
            "Aucun vin avec l'ensemble des critères renseignés n'a pu être trouvé.",
            icon="😵",
        )
    else:
        return None


def display_scatter(
    df: pl.DataFrame, selected_wines: list[str], colors: list[str], scale: str
) -> DeltaGenerator:
    """Génère un scatter plot du prix des vins avec plusieurs configurations."""
    if scale == "$\log(y)$":
        log = True
        title_y = "log(Prix unitaire)"
    elif scale == "$y$":
        log = False
        title_y = "Prix unitaire"
    warning = warnings(df, selected_wines)
    if not warning:
        scatter = px.scatter(
            df,
            x="conservation_time",
            y="unit_price",
            trendline="lowess",
            color="type",
            symbol="type",
            size="capacity",
            title=f"Prix d'un {' / '.join(selected_wines).lower()} en fonction de sa durée de conservation",
            hover_name="name",
            log_y=log,
            trendline_color_override="white",
            color_discrete_sequence=colors,
        )
        scatter.update_xaxes(title_text="Temps de conservation (en années)")
        scatter.update_yaxes(title_text=title_y, ticksuffix=" €", showgrid=True)
        st.plotly_chart(scatter)


def create_aggregate_df(df: pl.DataFrame) -> pl.DataFrame:
    """Crée un Dataframe groupé par pays et code ISO avec le nombre de vins."""
    grouped_df = (
        df.group_by("country", "iso_code")
        .count()
        .sort("count", descending=True)
        .filter(pl.col("country") != "12,5 % vol")
    )
    return grouped_df


def create_map(df: pl.DataFrame) -> DeltaGenerator:
    """Crée la carte de provenance des vins."""
    map = px.choropleth(
        df,
        locations="iso_code",
        hover_name="country",
        hover_data="count",
        color="country",
    )
    map.update_layout(
        geo_bgcolor="#0e1117",
        showlegend=False,
        margin=dict(l=20, r=20, t=0, b=0),
    )
    return st.plotly_chart(map)


def create_bar(grouped_df: pl.DataFrame) -> DeltaGenerator:
    """Crée un diagramme en barres du nombre de vins commercialisés par pays."""
    bar = px.bar(
        grouped_df,
        x="country",
        y="count",
        color_discrete_sequence=["white"],
        title="Nombre de vins commercialisés par pays",
        text="count",
    )
    bar.update_layout(margin=dict(l=20, r=20, t=25, b=0))
    bar.update_yaxes(visible=False)
    return st.plotly_chart(bar)


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


def write_table_ml(chemin_csv, mode):
    """Retourne un tableau avec les résultats des modèles"""
    df = pl.read_csv(chemin_csv)
    df = df.filter(df["Mode"] == mode)
    st.dataframe(
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


def display_corr(df: pl.DataFrame):
    """Retourne un plot de corrélation"""
    variables = [
        "capacity",
        "unit_price",
        "millesime",
        "avg_temp",
        "conservation_date",
        "bio",
        "customer_fav",
        "is_new",
        "top_100",
        "destock",
        "sulphite_free",
        "alcohol_volume",
        "bubbles",
    ]
    df_drop_nulls = df.select(variables).drop_nulls()
    cor_matrice = np.array(df_drop_nulls.corr())
    fig_corr = ff.create_annotated_heatmap(
        z=cor_matrice,
        x=variables,
        y=variables,
        annotation_text=np.around(np.array(df_drop_nulls.corr()), decimals=2),
        colorscale="Inferno",
    )
    masque = np.ma.masked_where(cor_matrice >= 0.99, cor_matrice)
    cor_min = round(np.min(masque), 2)
    cor_max = round(np.max(masque), 2)

    cor_min_txt = (
        f"➖ La corrélation minimale est de {cor_min} entre le millésime et le prix."
    )
    cor_max_txt = f"➕ La corrélation maximale est de {cor_max} entre la date de conservation et le prix."
    return (
        st.plotly_chart(fig_corr),
        st.success(cor_max_txt),
        st.error(cor_min_txt),
    )


def display_density(df: pl.DataFrame):
    """Retourne un plot de densité"""
    fig_tv = px.histogram(
        df,
        x="unit_price",
        marginal="box",
        nbins=4000,
        log_x=True,
        color="type",
        color_discrete_map={
            "Vin Rouge": "#ff4b4b",
            "Vin Blanc": "#f3b442",
            "Vin Rosé": "#ff8fa3",
        },
    )
    return st.plotly_chart(fig_tv)


def display_bar(df: pl.DataFrame):
    """Retourne un plot en bar"""
    cepage_counts = df.groupby("cepage").agg(pl.col("cepage").count().alias("count"))
    cepage_filtre = cepage_counts.filter(cepage_counts["count"] >= 10)
    df_filtre = df.join(cepage_filtre, on="cepage")
    fig_bar = px.bar(
        df_filtre,
        x="cepage",
        color="type",
        color_discrete_map={
            "Vin Rouge": "#ff4b4b",
            "Vin Blanc": "#f3b442",
            "Vin Rosé": "#ff8fa3",
        },
    )
    return st.plotly_chart(fig_bar)


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


def display_wine_img(df: pl.DataFrame, wine_name: str) -> DeltaGenerator:
    """Permet d'afficher l'image d'un vin prédit à partir de son nom."""
    link = (
        df.select("name", "picture")
        .filter(pl.col("name") == wine_name)
        .select("picture")
        .item()
    )
    return st.image(link, width=200)


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


##### Fonctions à bouger vers db_requests


def get_names(conn: DuckDBPyConnection) -> list[str]:
    """Récupère les noms des vins qui ont été prédits par le modèle."""
    result = conn.sql("SELECT name FROM pred_regression")
    names = [row[0] for row in result.fetchall()]
    return names