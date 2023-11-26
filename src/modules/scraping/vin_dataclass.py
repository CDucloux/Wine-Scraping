from dataclasses import dataclass
from serde import serialize


@serialize
@dataclass
class Vin:
    """Représente toutes les caractéristiques associées à un vin."""

    name: str
    capacity: str
    price: str
    price_bundle: str
    characteristics: str
    note: str
    keywords: list[str]
    others: str
    picture: str
    classification: str
    millesime: str
    cepage: str
    gouts: str
    par_gouts: str
    oeil: str
    nez: str
    bouche: str
    temperature: str
    service: str
    conservation_1: str
    conservation_2: str
    accords_vins: str
    accords_reco: str
