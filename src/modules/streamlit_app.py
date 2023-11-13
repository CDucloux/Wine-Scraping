import streamlit as st
import polars as pl
import time
import plotly.express as px
from bear_cleaner import *
from st_functions import *
from annotated_text import annotated_text, annotation


def main():
    page_config()
    remove_white_space()
    st.title("🍷 Vins à la carte")
    df = load_df()

    with st.sidebar:
        # Configure l'ensemble de la sidebar de paramètres
        st.header("*Paramètres*")
        with st.spinner("Chargement..."):
            time.sleep(0.75)
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

    # Génère un header avec des metrics, commun à toutes les pages
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
        st.write(
            "**TODO** : Etudier écarts types, corrélations, tests de Student, Inclure du latex, etc."
        )

    with tab3:
        colors = color_selector(selected_wines)
        # TODO: customiser les hover traces de plotly pour les rendre + sexy
        # TODO: Voir comment modifier pour laisser moins d'espace entre les buttons et plot
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
        ...

    with tab6:
        authors()
        # ajouter des images d'une cagette de vin because c'est sympa
        # voir pour ajouter un gradient sur une page aussi + fontawesome


if __name__ == "__main__":
    main()
