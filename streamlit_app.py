"""
`streamlit_app`
==============

Module principal orchestrant tous les composants de notre application streamlit :
- functions
- tables 
- selectors 
- plots
"""

import streamlit as st
import time
from src.modules.app.st_functions import *
from src.modules.app.st_tables import *
from src.modules.app.st_selectors import *
from src.modules.app.st_plots import *
from src.modules.utils import *


def main():
    page_config()
    remove_white_space()
    custom_radio_css()
    st.title("🍷 Vins à la carte")
    df = load_df()
    conn = db_connector()
    load_tables(conn)

    with st.sidebar:
        # Configure l'ensemble de la sidebar de paramètres
        st.image("img/wine_scraping_logo.png")
        st.header("*Paramètres*")
        with st.spinner("Chargement..."):
            time.sleep(0.25)
            selected_wines = sidebar_wine_selector()
            prices = sidebar_prices_slider(df)
            filter_bio = sidebar_checkbox_bio()
            filter_new = sidebar_checkbox_new()
            filter_fav = sidebar_checkbox_fav()
            user_input = sidebar_input_wine()

            main_df = load_main_df(
                df,
                selected_wines,
                prices,
                filter_bio,
                filter_new,
                filter_fav,
                user_input,
            )
            st.markdown(f">**{len(main_df)}** :red[vins] trouvés !")

    # Metrics vins
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            main_wine_metric(df, "Vin Rouge")
    with col2:
        with st.container(border=True):
            main_wine_metric(df, "Vin Blanc")
    with col3:
        with st.container(border=True):
            main_wine_metric(df, "Vin Rosé")
    st.write(
        f"**{len(df)}** vins récupérés grâce à un *crawler* sur https://www.vinatis.com/, explorons-les ! "
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📎 Data Overview",
            "🔍Statistiques Descriptives",
            "📈 Charts",
            "🌍 Provenance",
            "⚙ Machine Learning",
            "👨‍💻Auteurs",
        ]
    )

    with tab1:
        ## DataFrame & Exploration Tab
        warning = warn(main_df, selected_wines)
        if not warning:
            write_price(main_df, selected_wines)
            write_table(main_df)
            st.divider()

    with tab2:
        ## Stat Desc Tab
        info()
        choix = st.selectbox(
            "Que voulez-vous consulter ?",
            ("Histogramme des prix", "Matrice de corrélation", "Cépage majoritaire"),
        )

        if choix == "Matrice de corrélation":
            display_corr(df)
        if choix == "Histogramme des prix":
            display_density(df)
        if choix == "Cépage majoritaire":
            display_bar(df)
            st.warning(
                "🚨 Seuls les cépages ayant une fréquence supérieure à dix sont affichés !"
            )

    with tab3:
        ## Chart Tab
        colors = color_selector(selected_wines)
        warning = warn(main_df, selected_wines)
        if not warning:
            scale = scale_selector()
            display_scatter(main_df, selected_wines, colors, scale)

    with tab4:
        ## Provenance Tab
        warning = warn(main_df, selected_wines)
        with st.container(border=True):
            if not warning:
                grouped_df = create_aggregate_df(main_df)
                create_map(grouped_df)
                create_bar(grouped_df)

    with tab5:
        ## Machine Learning Tab
        info()
        st.markdown(
            "Pas vraiment adepte des concepts de *Machine Learning* ? **Pas de problème !** :link: [Cliquer ici pour prédire un vin](#a29a3caa)"
        )
        st.subheader("Exploration")
        choice = st.selectbox(
            "Choix des modèles de Machine Learning",
            (
                "Régression - Prédiction du prix",
                "Classification - Prédiction type de vin",
            ),
        )
        if choice == "Régression - Prédiction du prix":
            write_table_ml(conn, table_name="ml_regression")
        elif choice == "Classification - Prédiction type de vin":
            write_table_ml(conn, "ml_classification")
        st.markdown(
            """
        > Les scores de *Train CV* et *Test CV* affichés ici sont les résultats des meilleurs modèles issus d'une recherche par grille exhaustive à l'aide d'une **Validation Croisée**.
        """
        )
        st.divider()
        st.subheader("Investigation")
        st.markdown(
            "$\\Rightarrow$ Permet d'explorer les hyperparamètres optimisés et d'évaluer la qualité des modèles sur les données de Test avec différentes métriques !"
        )
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
            if choice == "Régression - Prédiction du prix":
                write_parameter(
                    conn, table_name="ml_regression", selected_model=selected_model
                )
                type = "regression"
            else:
                write_parameter(
                    conn, table_name="ml_classification", selected_model=selected_model
                )
                type = "classification"
        write_metrics(conn, type)
        with st.expander("**Importance des variables**"):
            n_vars = n_variable_selector()
            display_importance(conn, choice, selected_model, n_vars)
            st.write(
                "*Note* : L'importance des variables n'est disponible que dans les modèles de **Boosting** et de **Random Forest**."
            )
        if choice == "Classification - Prédiction type de vin":
            display_confusion_matrix(conn, model_mapper(selected_model))
        st.divider()
        st.subheader(":red[Prédiction]")
        names = get_names(conn)
        wine_name = st.selectbox("Vin : ", names)
        col1, col2 = st.columns([2, 2])
        with col1:
            display_wine_img(df, wine_name)
        with col2:
            choix_type = st.selectbox("Type :", ("Regression", "Classification"))
            best = best_model(choix_type, conn)
            model_choice = model_selector(best)
            model = model_mapper(model_choice)
            st.markdown(f"Modèle recommandé : **{best}**")
        col1, col2 = st.columns([2, 2])
        with col1:
            if choix_type == "Regression":
                truth = get_value(conn, "unit_price", "pred_regression", wine_name)
                pred = get_value(conn, model, "pred_regression", wine_name)
                st.metric(label="*Prix réel*", value=f"{truth} €".replace(".", ","))
            else:
                truth = get_value(conn, "type", "pred_classification", wine_name)
                pred = get_value(conn, model, "pred_classification", wine_name)
                st.metric(label="*Type de vin réel*", value=truth)
        with col2:
            if choix_type == "Regression":
                st.metric(
                    label="*Prix prédit* $^*$",
                    value=format_prediction(pred, truth),
                )
            else:
                st.metric(
                    label="*Type de vin prédit*", value=format_prediction(pred, truth)
                )
        if choix_type == "Regression":
            popover_prediction(pred, truth)
            with st.expander(
                "**Explications complémentaires sur les seuils de prix ⤵**"
            ):
                st.latex(
                    """
                    \\begin{cases}
                    price_{predicted} \\in [0.8\\cdot price_{true} \\hspace{0.1em}; 1.2\\cdot price_{true}] \\Rightarrow ✅ \\newline 
                    price_{predicted} \\notin [0.8\\cdot price_{true} \\hspace{0.1em}; 1.2\\cdot price_{true}] \\Rightarrow ❌
                    \\end{cases}
                    """
                )
                st.markdown(
                    "Autrement dit, quand le **prix prédit** est compris entre 80 et 120% du **prix réel**, alors la différence est considérée comme *acceptable*."
                )
        st.divider()

    with tab6:
        authors()


if __name__ == "__main__":
    main()
