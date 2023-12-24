from typing import Callable
from rich import print as rprint
from functools import wraps


def model_mapper_ml(model_name: str) -> str:
    """Mappe les noms des fonctions d'entrainement de `GridSearchCV` aux noms des modèles."""
    model_names_mapping = {
        "model_rf": "Random Forest",
        "model_boost": "Boosting",
        "model_ridge": "Ridge",
        "model_mlp": "Réseaux de neurones",
        "model_knn": "K Neighbors",
        "model_svm": "Support Vector",
    }
    return model_names_mapping.get(model_name, "Le modèle n'existe pas")


def model_name(func) -> Callable:
    """Permet à l'utilisateur de savoir quel modèle est entrainé dans le script d'entrainement."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        rprint(
            f"Entrainement du modèle : [bold]{model_mapper_ml(func.__name__)}[/bold]"
        )
        return func(*args, **kwargs)

    return wrapper
