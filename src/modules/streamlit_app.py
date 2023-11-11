import streamlit as st
import polars as pl
import numpy as np
from pathlib import Path
import time
import plotly.express as px
from bear_cleaner import *
from st_functions import *


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
        st.write(
            "**TODO** : Etudier écarts types, corrélations, tests de Student, Inclure du latex, etc."
        )

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
        if st.button("Clique ici fréro"):
            st.success("Mon fréro tu t'attendais à quoi mdrr ?")

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
