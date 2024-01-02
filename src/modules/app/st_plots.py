"""
`st_plots` : Le module qui génère les plots de l'application 📊
"""


import streamlit as st
import polars as pl
import numpy as np
from streamlit.delta_generator import DeltaGenerator
import plotly.express as px  # type: ignore
import plotly.figure_factory as ff  # type: ignore
import plotly.graph_objects as go  # type: ignore
from sklearn.metrics import confusion_matrix  # type: ignore
from duckdb import DuckDBPyConnection


def warn(df: pl.DataFrame, selected_wines: list[str]) -> DeltaGenerator | None:
    """`warn`: Renvoie des messages d'avertissements spécifiques quand le dataframe modifié
    à cause de la sidebar ne renvoie pas de données.

    ---------
    `Parameters`
    --------- ::

        df (pl.DataFrame): # DataFrame mutable
        selected_wines (list[str]): # Vin(s) sélectionné(s)

    `Returns`
    --------- ::

        DeltaGenerator | None

    `Example(s)`
    ---------

    >>> warn()
    ... #_test_return_"""
    if not selected_wines:
        return st.warning("🚨 Attention, aucun type de vin n'a été selectionné !")
    elif len(df) == 0:
        return st.warning(
            "😵 Aucun vin avec l'ensemble des critères renseignés n'a été trouvé."
        )
    else:
        return None


def display_scatter(
    df: pl.DataFrame, selected_wines: list[str], colors: list[str], scale: str
) -> DeltaGenerator:
    """`display_scatter`: Génère un scatter plot du prix des vins en fonction de leur
    durée de conservation avec plusieurs échelles disponibles.

    ---------
    `Parameters`
    --------- ::

        df (pl.DataFrame): # DataFrame mutable
        selected_wines (list[str]): # Vin(s) sélectionné(s)
        colors (list[str]): # Liste de couleurs
        scale (str): # Echelle (log/Linéaire)

    `Returns`
    --------- ::

        DeltaGenerator

    `Example(s)`
    ---------

    >>> display_scatter()
    ... #_test_return_"""
    if scale == "$\\log(y)$":
        log = True
        title_y = "log(Prix unitaire)"
    else:
        log = False
        title_y = "Prix unitaire"
    scatter = px.scatter(
        df,
        x="conservation_time",
        y="unit_price",
        trendline="lowess",
        color="type",
        size="capacity",
        title=f"Prix d'un {' / '.join(selected_wines).lower()} en fonction de sa durée de conservation",
        hover_name="name",
        log_y=log,
        trendline_color_override="white",
        color_discrete_sequence=colors,
    )
    scatter.update_xaxes(title_text="Temps de conservation (en années)")
    scatter.update_yaxes(title_text=title_y, ticksuffix=" €", showgrid=True)
    return st.plotly_chart(scatter)


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
        title="Carte de la provenance des vins",
    )
    map.update_layout(
        geo_bgcolor="#0e1117",
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=0),
    )
    return st.plotly_chart(map)


def create_bar(grouped_df: pl.DataFrame) -> DeltaGenerator:
    """Crée un diagramme en barres du nombre de vins commercialisés par pays."""
    bar = px.bar(
        grouped_df,
        x="country",
        y="count",
        color_discrete_sequence=["#ff4b4b"],
        title="Nombre de vins commercialisés par pays",
        text="count",
    )
    bar.update_traces(textfont_color="white")
    bar.update_layout(margin=dict(l=20, r=20, t=25, b=0))
    bar.update_yaxes(visible=False)
    bar.update_xaxes(title="Pays")
    return st.plotly_chart(bar)


def display_corr(
    df: pl.DataFrame,
) -> tuple[DeltaGenerator, DeltaGenerator, DeltaGenerator]:
    """`display_corr`: Retourne une matrice de corrélation avec corrélation minimale et maximale.

    ---------
    `Parameters`
    --------- ::

        df (pl.DataFrame): # DataFrame

    `Returns`
    --------- ::

        tuple[DeltaGenerator, DeltaGenerator, DeltaGenerator]

    `Example(s)`
    ---------

    >>> display_corr()
    ... #_test_return_"""
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
        colorscale="amp",
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


def display_density(df: pl.DataFrame) -> DeltaGenerator:
    """Retourne l'histogramme de densité des prix."""
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
    fig_tv.update_xaxes(title_text="Prix unitaire", ticksuffix=" €")
    fig_tv.update_yaxes(title_text="", showgrid=True)
    return st.plotly_chart(fig_tv)


def display_bar(df: pl.DataFrame) -> DeltaGenerator:
    """Retourne un barplot des cepages."""
    cepage_counts = df.groupby("cepage").count()
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
    fig_bar.update_yaxes(title_text="")
    return st.plotly_chart(fig_bar)


def display_wine_img(df: pl.DataFrame, wine_name: str) -> DeltaGenerator:
    """Permet d'afficher l'image d'un vin prédit à partir de son nom."""
    link = (
        df.select("name", "picture")
        .filter(pl.col("name") == wine_name)
        .select("picture")
        .item()
    )
    return st.image(link, width=200)


def display_confusion_matrix(conn: DuckDBPyConnection, model: str) -> DeltaGenerator:
    """Crée la matrice de confusion."""
    df = conn.execute(f"SELECT * FROM pred_classification").pl()
    y_true = df.select(pl.col("type"))
    y_pred = df.select(pl.col(model))
    conf_matrix = confusion_matrix(y_true, y_pred)
    labels = ["Vin Blanc", "Vin Rosé", "Vin Rouge"]

    cm_fig = px.imshow(
        conf_matrix,
        labels=dict(x="Prédictions", y="Réalité", color="Nombre"),
        x=labels,
        y=labels,
        color_continuous_scale="amp",
    )

    cm_fig.update_layout(title="Matrice de Confusion")

    for true_label, _ in enumerate(labels):
        for pred_label, _ in enumerate(labels):
            count = conf_matrix[pred_label][true_label]
            cm_fig.add_annotation(
                x=true_label,
                y=pred_label,
                text=str(count),
                showarrow=False,
                font=dict(color="black", size=12),
            )

    return st.plotly_chart(cm_fig)


def display_importance(
    conn: DuckDBPyConnection, choice: str, selected_model: str, n: int
) -> DeltaGenerator | None:
    """`display_importance`: Retourne un graphique montrant les 15 variables les plus importantes.

    - /❗\ Uniquement disponible pour les modèles à base d'arbres.

    ---------
    `Parameters`
    --------- ::

        choice (str)
        selected_model (str)

    `Returns`
    --------- ::

        DeltaGenerator | None

    `Example(s)`
    ---------"""
    if not selected_model in ("Random Forest", "Boosting"):
        return None

    if choice == "Régression - Prédiction du prix":
        target = "unit_price"
    else:
        target = "type"

    df = conn.execute(f"SELECT * FROM var_importance").pl()

    df_imp_model = df.filter(pl.col("id") == f"{target} {selected_model}")
    df_imp_tail = df_imp_model.tail(n)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_imp_tail.select("importances").to_series(),
            y=df_imp_tail.select("column_names").to_series(),
            orientation="h",
            marker=dict(color="#ff4b4b"),
        )
    )
    fig.update_layout(
        title=f"Importance relative des {n} variables les plus décisives",
        xaxis_title="Importance",
        yaxis_title="Variables",
        yaxis=dict(tickfont=dict(size=8)),
    )

    return st.plotly_chart(fig)
