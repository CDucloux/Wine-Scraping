# Wine-Scraping

- Faire un module qui permet de mesurer l'erreur empirique et l'erreur contrôle pour détecter éventuellement un surapprentissage du modèle et éviter qu'il prédise n'importe quoi.
- Récupérer `EQM` $\Rightarrow$ Erreur Quadratique Moyenne : 

```python
from sklearn.metrics import mean_squared_error
```

- [ ] En utilisant `cross_val_scores` on fait de la cross validation $\Rightarrow$ voir tp 6 pour l'implémentation
- [ ] Commencer à faire les tests unitaires et d'intégration et pytest coverage + doctest pour les tests dans les docstrings.
- [ ] Faire `poetry` et venv, voire `Docker`
- [ ] Regarder du coté de `MLFLOW` pour les métriques de Machine Learning

> Plan :

- **Scraping**
    - `scraping_functions` $\Rightarrow$ module finalisé
    - `scraper` $\Rightarrow$ module finalisé (récupère les hrefs des vins et les écrit dans un fichier csv)

- **Soup & JSON**
    - `mystical_soup` $\Rightarrow$ module finalisé (Transforme en les résultats trouvés dans les pages html à l'aide de *BeautifulSoup*)

- **Polars & Restructuration tabulaire**
    - `bear_cleaner` $\Rightarrow$ module finalisé (Transfomr le json brurt en format tabulaire exploitable)

- **Machine Learning**
    - `models` $\Rightarrow$ module finalisé (Prépare les modèles)
    - `prediction` $\Rightarrow$ module finalisé (Applique les modèles et fait les prédictions)

- **Base de données** : Alimentation d'une DB in memory suite au cleaning avec polars

- **Application** : Création d'une appli avec Dash ou Streamlit

- **Phase de tests unitaires, check MYPY, environnements virtuels, poetry, re-documentation & éventuellement Docker**

- [ ] Voir tests unitaires dans des docstrings $\Rightarrow$ `doctest`