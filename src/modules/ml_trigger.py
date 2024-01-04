"""Exécute les scripts qui permettent de créer les tables alimentant la base de données :

- optimisation 
- prédiction
- importance
"""

import subprocess

subprocess.call(["python", "-m", "src.modules.ml_models.optimisation_script"])
subprocess.call(["python", "-m", "src.modules.ml_models.prediction_script"])
subprocess.call(["python", "-m", "src.modules.ml_models.importance_script"])
