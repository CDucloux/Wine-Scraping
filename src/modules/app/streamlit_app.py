import streamlit as st
import time
from pathlib import Path

# from bear_cleaner import *
from st_functions import *
from src.modules.ml_models.prediction import *
import plotly.express as px


def main():
    page_config()
    remove_white_space()
    st.title("🍷 Vins à la carte")
    df = load_df()
    conn = db_connector()

    with st.sidebar:
        # Configure l'ensemble de la sidebar de paramètres
        st.header("*Paramètres*")
        with st.spinner("Chargement..."):
            time.sleep(0.25)
            selected_wines = sidebar_wine_selector()
            prices = sidebar_prices_slider(df)
            filter_bio = sidebar_checkbox_bio()
            filter_new = sidebar_checkbox_new()
            filter_fav = sidebar_checkbox_fav()
            user_input = sidebar_input_wine()
            years = sidebar_year_selector(df)

            main_df = load_main_df(
                df,
                selected_wines,
                prices,
                filter_bio,
                filter_new,
                filter_fav,
                user_input,
                years,
            )
            st.markdown(f">**{len(main_df)}** :red[vins] trouvés !")

    # Metrics vins
    col1, col2, col3 = st.columns(3)
    with col1:
        main_wine_metric(df, "Vin Rouge")
    with col2:
        main_wine_metric(df, "Vin Blanc")
    with col3:
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
        write_price(main_df, selected_wines)
        write_table(main_df)
        st.divider()

    with tab2:
        st.info(
            "L'ensemble de cet onglet est statique, la barre de paramètres n'influera pas sur les données.",
            icon="ℹ️",
        )
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

        # "**TODO** : Etudier écarts types, tests de Student, Inclure du latex, etc."

    with tab3:
        colors = color_selector(selected_wines)
        col4, col5 = st.columns([0.3, 0.7])
        with col4:
            scale = scale_selector()
            custom_radio_css()
        with col5:
            annotated_text(
                "This ",
                ("is", "Verb"),
                " some ",
                ("annotated", "Adj"),
                ("text", "Noun"),
                "And here's a ",
                ("word", ""),
                " with a fancy background but no label.",
            )
        display_scatter(main_df, selected_wines, colors, scale)

    with tab4:
        with st.container():
            grouped_df = create_aggregate_df(main_df)
            create_map(grouped_df)
            create_bar(grouped_df)

    with tab5:
        st.subheader("Exploration")
        choice = st.selectbox(
            "Choix des modèles de machine learning: ",
            (
                "Régression - Prédiction du prix",
                "Classification - Prédiction type de vin",
            ),
        )
        if choice == "Régression - Prédiction du prix":
            write_table_ml("./data/result_ml_regression.csv", "regression")
            st.divider()
            write_parameter("./data/result_ml_regression.csv", "regression")
        elif choice == "Classification - Prédiction type de vin":
            write_table_ml("./data/result_ml_classification.csv", "classification")
            st.divider()
            write_parameter("./data/result_ml_classification.csv", "classification")
        st.divider()
        st.subheader("Prédiction")
        names = get_names(conn)
        wine_name = st.selectbox("Vin : ", names)
        col1, col2 = st.columns([2, 2])
        with col1:
            display_wine_img(df, wine_name)
        with col2:
            choix_type = st.selectbox("Type :", ("Regression", "Classification"))
            model_choice = model_selector()
            model = model_mapper(model_choice)
        col1, col2 = st.columns([2, 2])
        with col1:
            if choix_type == "Regression":
                truth = conn.sql(
                    f"SELECT unit_price FROM pred_regression WHERE name = '{wine_name}'"
                ).fetchone()[0]
                pred = conn.sql(
                    f"SELECT {model} FROM pred_regression WHERE name = '{wine_name}'"
                ).fetchone()[0]
                st.metric(label="*Prix réel*", value=f"{truth} €".replace(".", ","))
            else:
                st.metric(label="*Type de vin réel*", value=20)
        with col2:
            if choix_type == "Regression":
                st.metric(
                    label="*Prix prédit* $^*$",
                    value=f"{round(pred,2)} €",
                    delta=f"{round(pred - truth, 2)} €",
                )
            else:
                st.metric(label="*Type de vin prédit*", value=20)
        st.caption(
            "$^*$ Il est possible que le prix prédit soit **très loin de la réalité**, voire même **négatif**, en dépit de nos efforts."
        )
        st.divider()
    with tab6:
        authors()
        # ajouter gradients


if __name__ == "__main__":
    main()
