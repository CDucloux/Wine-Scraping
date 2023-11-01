from dataclasses import dataclass
from serde import serialize


@serialize
@dataclass
class Vin:
    """This dataclass represents all characteristics associated with a Wine."""

    name: str
    capacity: float
    year: int
    price: str
    promo: str
    prix_promo: str
    note: float
    nb_avis: int
    type: str
    lien : str
    vol : int
    adjective : str
    cepage: str 

