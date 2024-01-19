"""
`st_selectors` : Module gérant les sélecteurs de l'application.
"""

import streamlit as st
import polars as pl


def sidebar_wine_selector() -> list[str]:
    """`sidebar_wine_selector`: Permet de sélectionner un type de vin.

    `Returns`
    --------- ::

        list[str]

    `Example(s)`
    ---------
    >>> sidebar_wine_selector()
    ... ['Vin Rouge']"""
    wine_types = ["Vin Blanc", "Vin Rouge", "Vin Rosé"]
    return st.multiselect(
        "Sélectionnez un type de vin :",
        wine_types,
        default="Vin Rouge",
        placeholder="Choisir un type de vin",
        key="wine_selector",
    )


def sidebar_prices_slider(df: pl.DataFrame) -> tuple[float, float]:
    """`sidebar_prices_slider`: Permet de choisir un intervalle de prix.

    ---------
    `Parameters`
    --------- ::

        df (pl.DataFrame)

    `Returns`
    --------- ::

        tuple[float, float]

    `Example(s)`
    ---------
    >>> df = load_df()
    >>> sidebar_prices_slider(df)
    ... (5.0, 400.0)"""
    return st.slider(
        "Sélectionnez un intervalle de prix :",
        df.select("unit_price").min().item(),
        df.select("unit_price").max().item(),
        (5.0, 400.0),
        format="%.2f€",
        key="price_slider",
    )


def sidebar_checkbox_bio() -> set[int]:
    """`sidebar_checkbox_bio`: Une case à cocher pour n'inclure que les vins bios.

    `Returns`
    --------- ::

        set[int]

    `Example(s)`
    ---------
    >>> sidebar_checkbox_bio()
    ... {0, 1}"""
    bio = st.sidebar.checkbox("N'inclure que les vins Bio 🌿")
    if bio:
        filter_bio = {1}
    else:
        filter_bio = {0, 1}
    return filter_bio


def sidebar_checkbox_new() -> set[int]:
    """`sidebar_checkbox_new`: Une case à cocher pour n'inclure que les nouveautés.

    `Returns`
    --------- ::

        set[int]

    `Example(s)`
    ---------
    >>> sidebar_checkbox_new()
    ... {0, 1}"""
    new = st.sidebar.checkbox("N'inclure que les nouveautés 🆕")
    if new:
        filter_new = {1}
    else:
        filter_new = {0, 1}
    return filter_new


def sidebar_checkbox_fav() -> set[int]:
    """`sidebar_checkbox_fav`: Une case à cocher pour n'inclure que les coups de coeur client.

    `Returns`
    --------- ::

        set[int]

    `Example(s)`
    ---------
    >>> sidebar_checkbox_fav()
    ... {0, 1}"""
    fav = st.sidebar.checkbox("N'inclure que les Coups de Coeur")
    if fav:
        filter_fav = {1}
    else:
        filter_fav = {0, 1}
    return filter_fav


def sidebar_input_wine() -> str:
    """`sidebar_input_wine`: Un user input permettant de rechercher un nom de vin.

    - L'utilisation de regex dans l'input est interdite car risque d'envoyer une erreur.

    `Returns`
    --------- ::

        str

    `Example(s)`
    ---------
    >>> sidebar_input_wine()
    ... ''"""
    user_input = st.text_input("Recherche par nom de vin :").upper()
    regex = ["*", "[", "+D", "{"]
    if user_input in regex:
        user_input = ""
    return user_input


def scale_selector() -> str | None:
    """`scale_selector`: Crée un radio button pour sélectionner l'échelle.

    `Returns`
    --------- ::

        str | None

    `Example(s)`
    ---------
    >>> scale_selector()
    ... '$y$'"""
    return st.radio(
        "Sélectionner une *échelle*",
        ["$y$", "$\\log(y)$"],
    )


def color_selector(selected_wines: list[str]) -> list[str]:
    """`color_selector`: Permet de choisir une couleur selon le vin sélectionné pour le scatter plot des vins.

    ---------
    `Parameters`
    --------- ::

    selected_wines (list[str])

    `Returns`
    --------- ::

        list[str]

    `Example(s)`
    ---------
    >>> color_selector(["Vin Rouge"])
    ... ['#ff4b4b']
    ---------
    >>> color_selector(["Vin Rouge", "Vin Blanc"])
    ... ['#f3b442', '#ff4b4b']"""
    red, white, pink = "#ff4b4b", "#f3b442", "#ff8fa3"
    if selected_wines == ["Vin Rouge"]:
        colors = [red]
    elif selected_wines == ["Vin Blanc"]:
        colors = [white]
    elif selected_wines == ["Vin Rosé"]:
        colors = [pink]
    elif selected_wines == ["Vin Rouge", "Vin Blanc"]:
        colors = [white, red]
    elif selected_wines == ["Vin Rouge", "Vin Rosé"]:
        colors = [red, pink]
    elif selected_wines == ["Vin Blanc", "Vin Rouge"]:
        colors = [white, red]
    elif selected_wines == ["Vin Blanc", "Vin Rosé"]:
        colors = [white, pink]
    elif selected_wines == ["Vin Rosé", "Vin Rouge"]:
        colors = [red, pink]
    elif selected_wines == ["Vin Rosé", "Vin Blanc"]:
        colors = [white, pink]
    elif selected_wines == ["Vin Rouge", "Vin Blanc", "Vin Rosé"]:
        colors = [white, red, pink]
    else:
        colors = [white, red, pink]
    return colors


def model_radio_selector() -> str | None:
    """`model_radio_selector`: Permet de sélectionner un modèle de Machine Learning avec des radio buttons.

    `Returns`
    --------- ::

        str | None

    `Example(s)`
    ---------
    >>> model_radio_selector()
    ... 'Boosting'"""
    return st.radio(
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


def model_selector() -> str | None:
    """`model_selector`: Permet de sélectionner un modèle de Machine learning avec une liste déroulante.

    `Returns`
    --------- ::

        str | None

    `Example(s)`
    ---------
    >>> model_selector()
    ... 'Random Forest'"""
    models = [
        "Random Forest",
        "Boosting",
        "Ridge",
        "Réseaux de neurones",
        "K Neighbors",
        "Support Vector",
    ]
    return st.selectbox(
        "Modèle :",
        (models),
    )


def n_variable_selector() -> int | float:
    """`n_variable_selector`: Permet à l'utilisateur de sélectionner le nombre des n variables à afficher dans le plot de l'importance des variables.

    - Valeur par défaut : 10
    - Valeur minimal : 2
    - Valeur maximal : 30

    `Returns`
    --------- ::

        int | float

    `Example(s)`
    ---------
    >>> n_variable_selector()
    ... 10"""
    min_vars = 2
    max_vars = 30
    default_vars = 10
    return st.number_input(
        "Nombre de variables à afficher :", min_vars, max_vars, default_vars
    )
