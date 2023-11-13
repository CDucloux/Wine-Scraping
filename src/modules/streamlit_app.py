import streamlit as st
import polars as pl
import numpy as np
from pathlib import Path
import time
import plotly.express as px
from bear_cleaner import *
from st_functions import *
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np


def main():
    page_config()
    remove_white_space()
    st.title("🍷 Vins à la carte")
    df = load_df()

    with st.sidebar:
        """Configure l'ensemble de la sidebar de paramètres"""
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

            # TODO: le main_df devrait etre déporté dans une fonction de st_functions
            # avec le décorateur @st.cache_data pour optimiser la vitesse d'exec
            main_df = (
                df.filter(pl.col("type").is_in(selected_wines))
                .filter(pl.col("unit_price") > prices[0])
                .filter(pl.col("unit_price") < prices[1])
                .filter(pl.col("bio").is_in(filter_bio))
                .filter(pl.col("is_new").is_in(filter_new))
                .filter(pl.col("customer_fav").is_in(filter_fav))
                .filter(pl.col("name").str.contains(user_input))
                .filter(pl.col("millesime").is_in(years))
            )
            st.markdown(f">**{len(main_df)}** :red[vins] trouvés !")

            if selected_wines == ["Vin Rouge"]:
                color = "#ff4b4b"
            elif selected_wines == ["Vin Blanc"]:
                color = "#f3b442"
            elif selected_wines == ["Vin Rosé"]:
                color = "#ff8fa3"
            else:
                color = "white"

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
            'Que voulez-vous consulter ?',
            ('Matrice de corrélation', 'Type de vin'))
        
        if choix == "Matrice de corrélation":
            variables, df_drop_nulls = corr_plot()
            fig_corr = ff.create_annotated_heatmap(
                z=np.array(df_drop_nulls.corr()),
                x=variables, y=variables,
                annotation_text = np.around(np.array(df_drop_nulls.corr()), decimals=2),
                colorscale='Cividis')
            fig_corr.update_layout(title_text='Matrice de corrélation')
            st.plotly_chart(fig_corr)
        if choix == "Type de vin":
            df_2 = load_df()
            fig_tv = px.histogram(
                df_2,
                x = "type"
            )
            fig_tv.update_layout(title_text="Effectifs par type de vin")
            fig_tv.update_xaxes(categoryorder="total descending")
            st.plotly_chart(fig_tv)
        #"**TODO** : Etudier écarts types, corrélations, tests de Student, Inclure du latex, etc."

    with tab3:
        st.write("TODO: customiser les traces de plotly pour les rendre + sexy")
        fig = px.scatter(
            main_df,
            x="conservation_time",
            y="unit_price",
            trendline="lowess",
            title=f"Prix d'un {selected_wines} en fonction de sa durée de conservation",
            hover_data=["name"],
            trendline_color_override="white",
            color_discrete_sequence=[color],
        )
        fig.update_xaxes(title_text="Temps de conservation (en années)")
        fig.update_yaxes(title_text="Prix unitaire (en €)")
        st.plotly_chart(fig)

    with tab4:
        grouped_df = (
            df.group_by("country", "iso_code").count().sort("count", descending=True)
        )
        map = px.choropleth(
            grouped_df,
            locations="iso_code",
            hover_name="country",
            hover_data="count",
            color="country",
        )
        map.update_layout(
            geo_bgcolor="#0e1117", showlegend=False, margin=dict(l=20, r=20, t=0, b=0)
        )
        st.plotly_chart(map)
        st.write("TODO : bar chart à rajouter avec le count par pays.")

    with tab5:
        st.subheader("Régression - Prédiction du prix")
        col1, col2 = st.columns([2,1.5])
        with col1:
            write_table_ml("./data/result_ml.csv")
        with col2:
            write_parameter("./data/result_ml.csv")

             
    with tab6:
        st.balloons()
        st.info("Licence CC-by-sa", icon="ℹ️")
        with st.expander("Découvrir les `auteurs` de l'application"):
            st.markdown(
                """
- *Corentin DUCLOUX* : https://github.com/CDucloux
- *Guillaume DEVANT* : https://github.com/devgui37
"""
            )
        # ajouter des images d'une cagette de vin because c'est sympa


if __name__ == "__main__":
    main()
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
