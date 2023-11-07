"""
In fair cyberspace, where we lay our scene,
A web page, rich with code, in hues unseen,
'Twas transformed by a tool of coding's grace,
BeautifulSoup, to parse in the browser's place.

With Python's might and tags of HTML,
This library did weave its magic spell,
A metamorphosis from text to soup,
From tangled web to structured code, in truth.
"""

# TODO: expliciter plus ce module.

from bs4 import BeautifulSoup as BS
from serde.json import to_json
from src.modules.vin_dataclass import Vin
from pathlib import Path


def _soupifier(page: str) -> BS:
    """Soupifie la page web."""
    soup = BS(page, "html.parser")
    return soup


def _scrap_name(soup: BS) -> str:
    """Recupère le nom du vin."""
    name = soup.find("h1", id="produit-titre").text.strip()
    return name


def _scrap_capacity(soup: BS) -> str | None:
    """Recupère la capacité de la bouteille."""
    class_attrs = [
        "contenance-v2 inline-block",
        "btn-sm bg-transparent color-gray border border-gray no-pointer inline-block",
    ]
    tags = ["div", "span"]

    # il y a deux classes possibles qui proviennent de deux tags html différents
    # dans le deuxième cas, il y a "3" capacités, mais en réalité = liste déroulante

    for tag, class_attr in zip(tags, class_attrs):
        capacity_element = soup.find(tag, class_=class_attr)
        if capacity_element:
            return capacity_element.text.strip()

    return None


def _scrap_price(soup: BS) -> str | None:
    """Récupère le prix de la bouteille à l'unité + les promos si il y en a."""
    try:
        price = soup.find("span", id="our_price_display").text.strip()
    except:
        price = None
    return price


def _scrap_price_bundle(soup: BS) -> str | None:
    """Récupère le prix des bouteilles par achat groupé."""
    try:
        price_bundle = soup.find(
            name="span", attrs={"id": "quantity_discount_pretaxe"}
        ).text.strip()
        eligible_qty = soup.find("meta", itemprop="eligibleQuantity").get("content")

        discount_per_qty = f"{price_bundle} par {eligible_qty}"
    except:
        discount_per_qty = None
    return discount_per_qty


def _scrap_characteristics(soup: BS) -> str | None:
    """Recupère les caractéristiques principales du vin."""
    characteristics = soup.find(
        name="span", attrs={"class": "no-padding-horizontal"}
    ).text
    return characteristics


def _scrap_notes(soup: BS) -> str | None:
    """Recupère la note et le nombre d'avis."""
    notes = soup.find(
        name="div", attrs={"class": "col-xs-12 padding-bottom-10"}
    ).text.strip()

    if not notes:
        return None
    return notes


def _scrap_keywords(soup: BS) -> list[str]:
    """Recupère les mots important mis en avant sur la page du produit."""
    kwd_class = "margin-right margin-bottom bg-gray-dark taille-md padding-horizontal-30 padding-vertical-5 rounded-corner-3 label"
    matches = soup.find_all(name="span", attrs={"class": kwd_class})
    keywords = [keyword.text for keyword in matches]
    return keywords


def _scrap_other(soup: BS) -> str | None:
    """Récupère d'autres attributs : bio, nouveauté, vigneron indépendant, etc."""
    other_characs = soup.find_all(
        "div",
        attrs={"class": "margin-top-3 display-inline"},
    )
    if not other_characs:
        return None

    other_characs = "/".join(
        [other_charac.text.strip() for other_charac in other_characs]
    )
    return other_characs


def _scrap_img(soup: BS) -> str:
    """Récupère le lien de l'image de la bouteille"""
    picture = soup.find(
        name="img", attrs={"class": "img-full-width img-max-450 center-block"}
    ).get("src")
    return picture


def _scrap_details(soup: BS) -> dict:
    """Crée un dictionnaire clé-valeur pour extraire des caractéristiques complémentaires."""
    key_class = "table-cell-css vertical-align-top padding-vertical-5 nowrap padding-right-10 taille-xs color-gray-darker"
    value_class = "table-cell-css vertical-align-top padding-vertical-5 taille-xs color-gray-darker text-bold"
    keys = soup.find_all("div", attrs={"class": key_class})
    values = soup.find_all("div", attrs={"class": value_class})
    dict_details = dict(
        [(key.text.strip(), value.text.strip()) for key, value in zip(keys, values)]
    )
    return dict_details


def _get_value(dict_details: dict, key: str) -> str | None:
    """Récupère dans le dictionnaire de détails la valeur associée à une clé existante."""
    try:
        value = dict_details[key]
    except KeyError:
        value = None
    return value


def scraping(page: str) -> Vin:
    """Agrégateur de tous les fonctions de scraping"""
    soup = _soupifier(page)
    dict_details = _scrap_details(soup)

    results = Vin(
        name=_scrap_name(soup),
        capacity=_scrap_capacity(soup),
        price=_scrap_price(soup),
        price_bundle=_scrap_price_bundle(soup),
        characteristics=_scrap_characteristics(soup),
        note=_scrap_notes(soup),
        keywords=_scrap_keywords(soup),
        classification=_get_value(dict_details, "Classification"),
        millesime=_get_value(dict_details, "Millésime"),
        cepage=_get_value(dict_details, "Cépage"),
        gouts=_get_value(dict_details, "Goûts"),
        par_gouts=_get_value(dict_details, "Par Goûts"),
        oeil=_get_value(dict_details, "A l'oeil"),
        nez=_get_value(dict_details, "Au nez"),
        bouche=_get_value(dict_details, "En bouche"),
        temperature=_get_value(dict_details, "Température de service"),
        service=_get_value(dict_details, "Service"),
        conservation_1=_get_value(dict_details, "Jusqu'à"),
        conservation_2=_get_value(dict_details, "Conservation"),
        accords_vins=_get_value(dict_details, "Accords mets-vin"),
        accords_reco=_get_value(dict_details, "Accords recommandés"),
        others=_scrap_other(soup),
        picture=_scrap_img(soup),
    )

    return results


def create_json(all_pages: list[str]) -> None:
    """Crée un fichier semi-structuré JSON."""
    root = Path(".").resolve()
    data_folder = root / "data"

    vins = list()
    for page in all_pages:
        vins.append(scraping(page))

    vins_json = to_json(vins)
    file_path = data_folder / "vins.json"

    with open(file_path, "w", encoding="utf-8") as json_file:
        json_file.write(vins_json)

    return print(f"Export en JSON réalisé avec succès dans {data_folder} !")
