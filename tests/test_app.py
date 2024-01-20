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
    """L'utilisateur ne sélectionne aucun vin 
    dans la sidebar, résultant en un message d'avertissement."""
    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    app.sidebar.multiselect(key="wine_selector").set_value([]).run(timeout=15)
    assert (
        app.tabs[0].warning[0].value
        == "🚨 Attention, aucun type de vin n'a été selectionné !"
    )


def test_multiselect_vin_blanc_nombre():
    """L'utilisateur ne sélectionne que les vins
    blancs dans la sidebar. 1311 vins en résultent."""
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
    """L'utilisateur sélectionne l'onglet statistiques 
    descriptives et regarde la matrice des corrélations."""
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
    """L'utilisateur sélectionne l'onglet statistiques 
    descriptives et regarde les cépages majoritaires."""
    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    app.tabs[1].selectbox[0].set_value("Cépage majoritaire").run(timeout=15)
    assert (
        app.tabs[1].warning.values[0]
        == "🚨 Seuls les cépages ayant une fréquence supérieure à dix sont affichés !"
    )


def test_score_train_CV_reg():
    """
    L'utilisateur sélectionne l'onglet Machine Learning et le type prédiction du prix.
    Il consulte ensuite les scores d'entrainement de la Cross Validation.
    """
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


def test_score_train_CV_classif():
    """
    L'utilisateur sélectionne l'onglet Machine Learning
    et le type classification du type de vin.
    Il consulte ensuite les scores d'entrainement de la Cross Validation.
    """
    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    app.tabs[4].selectbox[0].set_value("Classification - Prédiction type de vin").run(
        timeout=15
    )
    assert app.tabs[4].dataframe.values[0]["Score Entrainement"].tolist() == [
        0.941,
        0.965,
        0.997,
        1.0,
        0.983,
        0.992,
    ]


def test_score_test_CV_reg():
    """
    L'utilisateur sélectionne l'onglet Machine Learning et le type prédiction du prix.
    Il consulte ensuite les scores de test de la Cross Validation.
    """
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


def test_score_test_CV_classif():
    """
    L'utilisateur sélectionne l'onglet Machine Learning
    et le type classification du type de vin.
    Il consulte ensuite les scores de test de la Cross Validation.
    """
    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    app.tabs[4].selectbox[0].set_value("Classification - Prédiction type de vin").run(
        timeout=15
    )
    assert app.tabs[4].dataframe.values[0]["Score Test"].tolist() == [
        0.934,
        0.954,
        0.976,
        0.975,
        0.979,
        0.981,
    ]


def test_hyperparams_reg():
    """
    L'utilisateur sélectionne l'onglet Machine Learning,
    le type prédiction du prix et consulte les hyperparamètres 
    optimaux du modèle Random Forest.
    """
    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    app.tabs[4].selectbox[0].set_value("Régression - Prédiction du prix").run(
        timeout=15
    )
    app.tabs[4].radio[0].set_value("Random Forest").run(timeout=15)
    assert app.tabs[4].dataframe.values[1]["Paramètres ⚒️"].tolist() == [
        "Profondeur Maximale",
        "N estimators",
        "Stratégie d'imputation",
    ]
    assert app.tabs[4].dataframe.values[1]["Valeur optimale ⭐"].tolist() == [
        "9",
        "40",
        "mean",
    ]


def test_hyperparams_classif():
    """
    L'utilisateur sélectionne l'onglet Machine Learning,
    le type prédiction de classification du type de vin,
    et consulte les hyperparamètres optimaux du modèle MLP.
    """
    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    app.tabs[4].selectbox[0].set_value("Classification - Prédiction type de vin").run(
        timeout=15
    )
    app.tabs[4].radio[0].set_value("Réseaux de neurones").run(timeout=15)
    assert app.tabs[4].dataframe.values[1]["Paramètres ⚒️"].tolist() == [
        "Hidden Layer Size",
        "Itération maximale",
        "Solveur",
        "Stratégie d'imputation",
    ]
    assert app.tabs[4].dataframe.values[1]["Valeur optimale ⭐"].tolist() == [
        "(100,)",
        "1000",
        "adam",
        "median",
    ]


def test_ml_metrics_reg():
    """
    L'utilisateur sélectionne l'onglet Machine Learning,
    le type prédiction de prédiction du prix,
    et consulte quelques métriques des modèles sur l'ensemble de test.
    """
    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    app.tabs[4].selectbox[0].set_value("Régression - Prédiction du prix").run(
        timeout=15
    )
    assert app.tabs[4].dataframe.values[2]["Mean Absolute Error ❗"].tolist() == [
        23.1,
        23.5,
        29.2,
        25.8,
        29.6,
        21.5,
        2340618513.1,
    ]


def test_ml_metrics_classif():
    """
    L'utilisateur sélectionne l'onglet Machine Learning,
    le type prédiction de classification du type de vin,
    et consulte quelques métriques des modèles sur l'ensemble de test.
    """
    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    app.tabs[4].selectbox[0].set_value("Classification - Prédiction type de vin").run(
        timeout=15
    )
    assert app.tabs[4].dataframe.values[2]["Accuracy Score 🏹"].tolist() == [
        0.936,
        0.98,
        0.981,
        0.958,
        0.984,
        0.983,
        0.938,
    ]
    assert app.tabs[4].dataframe.values[2]["F1-Score 🛠️"].tolist() == [
        0.926,
        0.98,
        0.981,
        0.957,
        0.984,
        0.983,
        0.933,
    ]


def truth_vin_tempranillo_reg_svm():
    """
    L'utilisateur sélectionne l'onglet Machine Learning,
    le type prédiction de prix,
    utilise un modèle SVM et consulte le prix réel d'un vin.
    """
    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    app.tabs[4].selectbox[1].set_value("TEMPRANILLO 2021- VEGA DEMARA").run(timeout=15)
    app.tabs[4].selectbox[2].set_value("Regression").run(timeout=15)
    app.tabs[4].selectbox[3].set_value("Support Vector").run(timeout=15)
    assert app.tabs[4].metric[0].value == "5,9 €"


def pred_vin_tempranillo_reg_svm():
    """
    L'utilisateur sélectionne l'onglet Machine Learning,
    le type prédiction de prix,
    utilise un modèle SVM et consulte le prix prédit d'un vin.
    """
    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    app.tabs[4].selectbox[1].set_value("TEMPRANILLO 2021- VEGA DEMARA").run(timeout=15)
    app.tabs[4].selectbox[2].set_value("Regression").run(timeout=15)
    app.tabs[4].selectbox[3].set_value("Support Vector").run(timeout=15)
    assert app.tabs[4].metric[1].value == "✅ 5,94 €"
    assert (
        app.tabs[4].error[0].value
        == "✔ Le prix prédit est 0,04 € **supérieur** au prix réel, soit une différence acceptable !"
    )
