from bs4 import BeautifulSoup as BS
import re
from src.modules.vin_dataclass import Vin


def extract_name(text: str) -> str:
    """
    Extract the `name` of the wine from unstructured text.
    """
    name = text.replace("\xa0", "_").split("_")[0]
    return name


def extract_capacity(text: str) -> float:
    """
    Extract the `capacity` of the wine from unstructured text.
    """
    capacity = float(
        text.replace("\xa0", "_")[
            text.replace("\xa0", "_").find("_") + 1 :
        ]  # remplacement des valeurs xa0 par _
        .replace("L", "")  # remplacement de l'indicateur litres "L" par rien
        .strip()  # on retire ensuite les caractères vides
    )
    return capacity


def extract_year(text: str) -> int | None:
    """
    Extract the `year` of the wine from unstructured text using a regular expression pattern.
    """
    match = re.search(r"\b\d{4}\b", text)
    if match:
        return int(match.group())
    else:
        return None


# TODO: Améliorer le motif pour trouver le "vrai" prix - outliers à 115-120 € à cause de la vente groupée


def extract_price(text: str) -> float | None:
    match = re.search(r"\d+,\d+", text)
    if match:
        return float(match.group().replace(",", "."))
    else:
        return None


def extract_promo(text: str) -> str | None:
    match = re.search(r"-\d+%", text)
    if match:
        return match.group()
    else:
        return None


def extract_price_promo(text: str) -> float | None:
    match = re.search(r"€(\d+,\d+)", text)
    if match:
        return float(match.group(1).replace(",", "."))
    else:
        return None


def extract_note(text: str) -> float | None:
    """On regarde uniquement text[:10] car sinon les dates font chier le motif de regex..."""
    match = re.search(
        r"([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?/20", text[:10]
    )
    if match:
        return float(match.group(1))
    else:
        return None


def extract_nb_avis(text: str) -> int:
    match = re.search(r"(\d+) notes", text)
    if match:
        return int(match.group(1))
    else:
        return 0


def extract_type(text: str) -> str:
    type = text.split(":")[0].strip()[text.split(":")[0].find("VIN") :]
    return type

def extract_link(result_title_notxt : BS):
    lien = result_title_notxt.find('a').get('href')
    return "https://www.vinatis.com" + lien


# TODO : resulset_i pas vraiment explicite dans la fonction extract_vin. Pas ouf.


def extract_vin(resultset_1, resultset_2, resultset_3, resultset_4, resultset_5) -> Vin:
    name = extract_name(resultset_1)
    capacity = extract_capacity(resultset_1)
    year = extract_year(resultset_1)
    price = extract_price(resultset_2)
    promo = extract_promo(resultset_2)
    prix_promo = extract_price_promo(resultset_2)
    note = extract_note(resultset_3)
    nb_avis = extract_nb_avis(resultset_3)
    type = extract_type(resultset_4)
    lien = extract_link(resultset_5)
    vol = None
    adjective = None
    cepage = None

    return Vin(
        name=name,
        capacity=capacity,
        year=year,
        price=price,
        promo=promo,
        prix_promo=prix_promo,
        note=note,
        nb_avis=nb_avis,
        type=type,
        lien=lien,
        vol=vol,
        adjective=adjective,
        cepage=cepage
    )


# A expliciter.


def extract_result(soupe: BS) -> dict:
    result_find_title = soupe.find_all(name="div", attrs={"class": "vue-product-name"})
    result_title = [x.text for x in result_find_title]
    result_title_notxt = [x for x in result_find_title]
    result_find_avis = soupe.find_all(name="div", attrs={"class": "vue-avis-block"})
    result_avis = [x.text for x in result_find_avis]
    result_find_price = soupe.find_all(
        name="div", attrs={"class": "vue-product-prices"}
    )
    result_price = [x.text for x in result_find_price]
    result_find_type = soupe.find_all(name="title")
    result_type = [x.text for x in result_find_type] * len(result_title)
    # comme il n'y a qu'un seul résultat pour le type de vin on multiplie par len(result_title)
    return {
        "result_title": result_title,
        "result_price": result_price,
        "result_avis": result_avis,
        "result_type": result_type,
        "result_title_notxt" : result_title_notxt
    }
    