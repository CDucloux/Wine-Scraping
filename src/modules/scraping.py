"""Module de scraping d'une page d'un produit sur le site vinatis.com"""
import requests as rq
from bs4 import BeautifulSoup as BS
from serde.json import to_json
from src.modules.scraping_class import Vin


def _scrap_soupe(adresse: str) -> BS:
    """Recupère la soupe"""
    requete = rq.get(url=adresse, timeout=10)
    soupe = BS(requete.text)
    return soupe


def _scrap_name(soupe: BS) -> str:
    """Recupère le nom du vin"""
    name = soupe.find_all(name="span", attrs={"itemprop": "name"})[0].text
    return name


def _scrap_capacity(soupe: BS) -> str:
    """Recupère la capacité de la bouteille"""
    capacity = soupe.find_all(name="span", attrs={"inline-block"})[0].text
    return capacity


def _scrap_price(soupe: BS) -> str:
    """Recupère les prix (prix de base, prix promo n°1, prix promo n°2)"""
    price = soupe.find_all(name="span", attrs={"id": "our_price_display"})[0].text

    try:
        price_discount_1 = soupe.find_all(
            name="span", attrs={"id": "quantity_discount_pretaxe"}
        )[0].text
    except:
        price_discount_1 = "None"

    try:
        price_discount_2 = soupe.find_all(
            name="span", attrs={"id": "quantity_discount_pretaxe"}
        )[1].text
    except:
        price_discount_2 = "None"
    return price, price_discount_1, price_discount_2


def _scrap_characteristic(soupe: BS) -> str:
    """Recupère les caractéristiques principal du produit"""
    characteristic = soupe.find_all(
        name="span", attrs={"class": "no-padding-horizontal"}
    )[0].text
    return characteristic


def _scrap_note(soupe: BS) -> str:
    """Recupère la note et le nombre d'avis"""
    note = soupe.find_all(name="div", attrs={"class": "col-xs-12 padding-bottom-10"})[
        0
    ].text
    return note


def _scrap_keyword(soupe: BS) -> list:
    """Recupère les mots important mis en avant sur la page du produit"""
    keyword = []
    class_k = "margin-right margin-bottom bg-gray-dark taille-md padding-horizontal-30 padding-vertical-5 rounded-corner-3 label"
    result = soupe.find_all(name="span", attrs={"class": class_k})
    for res in result:
        keyword.append(res.text)
    return keyword


def _scrap_informations(soupe: BS) -> str:
    """Recupère tous les élements qui détails le produit"""
    class_n = "table-cell-css vertical-align-top padding-vertical-5 nowrap padding-right-10 taille-xs color-gray-darker"
    class_value = "table-cell-css vertical-align-top padding-vertical-5 taille-xs color-gray-darker text-bold"

    details = [
        " \xa0\xa0Classification vin",
        " \xa0\xa0Millésime",
        " \xa0\xa0Cépage",
        " \xa0\xa0Goûts",
        " \xa0\xa0Par Goûts",
        " \xa0\xa0A l'oeil",
        " \xa0\xa0Au nez",
        " \xa0\xa0En bouche",
        " \xa0\xa0Température de service",
        " \xa0\xa0Service",
        " \xa0\xa0Conservation",
        " \xa0\xa0Jusqu'à",
        " \xa0\xa0Accords mets-vin",
        " \xa0\xa0Accords recommandés",
    ]

    dictionnaire_vin = {
        " \xa0\xa0Classification vin": [],
        " \xa0\xa0Millésime": [],
        " \xa0\xa0Cépage": [],
        " \xa0\xa0Goûts": [],
        " \xa0\xa0Par Goûts": [],
        " \xa0\xa0A l'oeil": [],
        " \xa0\xa0Au nez": [],
        " \xa0\xa0En bouche": [],
        " \xa0\xa0Température de service": [],
        " \xa0\xa0Service": [],
        " \xa0\xa0Conservation": [],
        " \xa0\xa0Jusqu'à": [],
        " \xa0\xa0Accords mets-vin": [],
        " \xa0\xa0Accords recommandés": [],
    }

    place_element = soupe.find_all(name="div", attrs={"class": class_n})
    value_element = soupe.find_all(name="div", attrs={"class": class_value})
    details_page = [place_element[i].text for i in range(len(place_element))]
    value_page = [value_element[i].text for i in range(len(value_element))]

    j = 0
    for i, detail in zip(range(14), details):
        try:
            details_page.index(details[i])
            dictionnaire_vin[detail] = value_page[j]
            j = j + 1
        except:
            dictionnaire_vin[detail] = "None"

    return dictionnaire_vin


def _scrap_img(soupe: BS) -> str:
    """Récupère le lien de l'image de la bouteille"""
    picture = soupe.find(
        name="img", attrs={"class": "img-full-width img-max-450 center-block"}
    ).get("src")
    return picture


def scraping(adresse):
    """Agrégateur de tous les fonctions de scraping"""
    soupe = _scrap_soupe(adresse)

    results = Vin(
        _scrap_name(soupe),
        _scrap_capacity(soupe),
        _scrap_price(soupe),
        _scrap_characteristic(soupe),
        _scrap_note(soupe),
        _scrap_keyword(soupe),
        _scrap_informations(soupe),
        _scrap_img(soupe),
    )

    return results


def json_brut(vins):
    """Créer un json"""
    vins_json = to_json(vins)

    file_path = "data/vins.json"

    with open(file_path, "w", encoding="utf-8") as json_file:
        json_file.write(vins_json)
