from dataclasses import dataclass
from serde import serialize

@serialize
@dataclass
class Vin:
    """This dataclass represents all characteristics associated with a Wine."""

    name: str
    capacity: str
    prices : str
    characteristic : str
    note: str
    keyword : str
    informations : str
    picture : str