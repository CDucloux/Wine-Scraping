# Wine-Scraping

J'ai continué un peu le webscrapping, tu peux pour le moment le voir dans ma branche "guillaume".

- Faire un module qui permet de mesurer l'erreur empirique et l'erreur contrôle pour détecter éventuellement un surapprentissage du modèle et éviter qu'il prédise n'importe quoi.
- Récupérer `EQM` $\Rightarrow$ Erreur Quadratique Moyenne : 

```python
from sklearn.metrics import mean_squared_error
```

- [ ] En utilisant `cross_val_scores` on fait de la cross validation $\Rightarrow$ voir tp 6 pour l'implémentation
- [ ] Voir tests unitaires dans des docstrings `doctest`
- [ ] Si try/except blocks, utiliser try/except/else

Plan :

Récupération de tous les href
- ...

Récupération des données brut
-scraping.py
-scraping_class.py

Nettoyage données brut
- ...
