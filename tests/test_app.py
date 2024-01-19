from streamlit.testing.v1 import AppTest


def test_metric_red():
    """Teste le format de la card associée aux vins rouges."""
    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    assert app.main.metric[0].label == ":red[Vin Rouge]"
    assert app.main.metric[0].value == "2479"
    assert app.main.metric[0].delta == "221 nouveautés !"


def test_metric_white():
    """Teste le format de la card associée aux vins blancs."""
    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    assert app.main.metric[1].label == ":orange[Vin Blanc]"
    assert app.main.metric[1].value == "1322"
    assert app.main.metric[1].delta == "119 nouveautés !"


def test_metric_pink():
    """Teste le format de la card associée aux vins rosés."""
    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    assert app.main.metric[2].label == "Vin Rosé"
    assert app.main.metric[2].value == "205"
    assert app.main.metric[2].delta == "2 nouveautés !"


def test_multiselect_none():
    """L'utilisateur ne sélectionne aucun vin dans la sidebar. 0 vins en résultent."""
    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    app.sidebar.multiselect(key="wine_selector").set_value([]).run(timeout=15)
    assert app.sidebar.markdown[0].value == ">**0** :red[vins] trouvés !"


def test_multiselect_none_warning():
    """L'utilisateur ne sélectionne aucun vin dans la sidebar, résultant en un message d'avertissement."""
    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    app.sidebar.multiselect(key="wine_selector").set_value([]).run(timeout=15)
    assert (
        app.tabs[0].warning[0].value
        == "🚨 Attention, aucun type de vin n'a été selectionné !"
    )


def test_multiselect_vin_blanc_nombre():
    """L'utilisateur ne sélectionne que les vins blancs dans la sidebar. 1311 vins en résultent."""
    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    app.sidebar.multiselect(key="wine_selector").set_value(["Vin Blanc"]).run(
        timeout=15
    )
    assert app.sidebar.markdown[0].value == ">**1311** :red[vins] trouvés !"


def test_multiselect_vin_blanc_prix_moyen():
    """L'utilisateur ne sélectionne que les vins blancs dans la sidebar."""
    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    app.sidebar.multiselect(key="wine_selector").set_value(["Vin Blanc"]).run(
        timeout=15
    )
    assert (
        app.tabs[0].markdown[0].value
        == "Le prix moyen d'un vin blanc  de la sélection est de  `29,58 €`."
    )


def test_price_slider():
    """L'utilisateur sélectionne un intervalle de prix restreint."""
    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    app.sidebar.slider("price_slider").set_range(740, 990).run(timeout=15)
    assert (
        app.tabs[0].markdown[0].value
        == "Le prix moyen d'un vin rouge  de la sélection est de  `799,4 €`."
    )


def test_price_slider_data():
    """L'utilisateur sélectionne un intervalle de prix restreint.
    - Le test compare les noms des vins du dataframe résultant.
    """
    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    app.sidebar.slider("price_slider").set_range(740, 990).run(timeout=15)
    assert app.tabs[0].dataframe.values[0]["name"].tolist() == [
        "MATHUSALEM - CORTON POUGETS 2013 - LOUIS JADOT - CAISSE BOIS",
        "CHATEAU LATOUR 2015 - PREMIER CRU CLASSE",
        "MAYA 2019 - DALLA VALLE VINEYARDS",
        "MAGNUM - HOMMAGE A JACQUES PERRIN 2020 - CHATEAU DE BEAUCASTEL",
        "CHÂTEAU MOUTON ROTHSCHILD 2001 - 1ER CRU CLASSÉ",
    ]


def test_correlations():
    """L'utilisateur sélectionne l'onglet statistiques descriptives et regarde la matrice des corrélations."""
    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    app.tabs[1].selectbox[0].set_value("Matrice de corrélation").run(timeout=15)
    assert (
        app.tabs[1].success.values[0]
        == "➕ La corrélation maximale est de 0.52 entre la date de conservation et le prix."
    )
    assert (
        app.tabs[1].error.values[0]
        == "➖ La corrélation minimale est de -0.38 entre le millésime et le prix."
    )


def test_cepages():
    """L'utilisateur sélectionne l'onglet statistiques descriptives et regarde les cépages majoritaires."""
    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    app.tabs[1].selectbox[0].set_value("Cépage majoritaire").run(timeout=15)
    assert (
        app.tabs[1].warning.values[0]
        == "🚨 Seuls les cépages ayant une fréquence supérieure à dix sont affichés !"
    )


def test_score_train_CV_reg():
    """L'utilisateur sélectionne l'onglet Machine Learning et le type prédiction du prix."""
    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    app.tabs[4].selectbox[0].set_value("Régression - Prédiction du prix").run(
        timeout=15
    )
    assert app.tabs[4].dataframe.values[0]["Score Entrainement"].tolist() == [
        0.82,
        0.398,
        0.619,
        0.781,
        0.423,
        0.63,
    ]


def test_score_test_CV_reg():
    """L'utilisateur sélectionne l'onglet Machine Learning et le type prédiction du prix."""
    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    app.tabs[4].selectbox[0].set_value("Régression - Prédiction du prix").run(
        timeout=15
    )
    assert app.tabs[4].dataframe.values[0]["Score Test"].tolist() == [
        0.398,
        0.286,
        0.348,
        0.427,
        0.368,
        0.38,
    ]
