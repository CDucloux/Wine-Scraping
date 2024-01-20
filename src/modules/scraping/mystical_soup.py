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

from bs4 import BeautifulSoup as BS, Tag
from serde.json import to_json
from src.modules.scraping.vin_dataclass import Vin  # type: ignore
from pathlib import Path
import requests as rq

def _soupifier(page: str) -> BS:
    """`_soupifier`: Soupifie la page web.

    ---------
    `Parameters`
    --------- ::

        page (str): # Page HTML 

    `Returns`
    --------- ::

        BS (BeautifulSoup)

    `Example(s)`
    --------- ::
    >>> page = requests.get(
        'https://www.vinatis.com/58600-chinon-les-picasses-2017-domaine-olga-raffault')
    >>> _soupifier(page.text)
    ... <!DOCTYPE HTML> {CONTENU DE LA PAGE} </html>"""
    soup = BS(page, "html.parser")
    return soup


def _scrap_name(soup: BS) -> str | None:
    """`_scrap_name`: Recupère le nom du vin.

    ---------
    `Parameters`
    --------- ::

        soup (BS): 

    `Returns`
    --------- ::

        str 

    `Example(s)`
    --------- ::
    >>> page = requests.get(
        'https://www.vinatis.com/58600-chinon-les-picasses-2017-domaine-olga-raffault')
    >>> soupe = _soupifier(page.text)
    >>> _scrap_name(soupe)
    ... 'CHINON LES PICASSES 2017 - DOMAINE OLGA RAFFAULT'"""
    name = soup.find("h1", id="produit-titre")

    if isinstance(name, Tag):
        return name.text.strip()
    else:
        return None


def _scrap_capacity(soup: BS) -> str | None:
    """`_scrap_capacity`: Recupère la capacité de la bouteille.

    ---------
    `Parameters`
    --------- ::

        soup (BS): 

    `Returns`
    --------- ::

        str | None

    `Example(s)`
    --------- ::
    >>> page = requests.get(
        'https://www.vinatis.com/58600-chinon-les-picasses-2017-domaine-olga-raffault')
    >>> soupe = _soupifier(page.text)
    >>> _scrap_capacity(soupe)
    ... '0,75 L'"""
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
    """`_scrap_price`: Récupère le prix de la bouteille à l'unité
    et les promos si il y en a.

    ---------
    `Parameters`
    --------- ::

        soup (BS): 

    `Returns`
    --------- ::

        str | None

    `Example(s)`
    --------- ::
    >>> page = requests.get(
        'https://www.vinatis.com/58600-chinon-les-picasses-2017-domaine-olga-raffault')
    >>> soupe = _soupifier(page.text)
    >>> _scrap_price(soupe)
    ... '15,00 €'"""
    price = soup.find("span", id="our_price_display")

    if isinstance(price, Tag):
        return price.text.strip()
    else:
        return None


def _scrap_price_bundle(soup: BS) -> str | None:
    """`_scrap_price_bundle`: Récupère le prix des bouteilles par achat groupé.

    ---------
    `Parameters`
    --------- ::

        soup (BS): 

    `Returns`
    --------- ::

        str | None

    `Example(s)`
    --------- ::
    >>> page = requests.get(
        'https://www.vinatis.com/58600-chinon-les-picasses-2017-domaine-olga-raffault')
    >>> soupe = _soupifier(page.text)
    >>> _scrap_price(soupe)
    ... null"""
    price_bundle_matches = soup.find(
        name="span", attrs={"id": "quantity_discount_pretaxe"}
    )

    eligible_qty_matches = soup.find("meta", itemprop="eligibleQuantity")

    if isinstance(price_bundle_matches, Tag):
        price_bundle = price_bundle_matches.text.strip()

    if isinstance(eligible_qty_matches, Tag):
        eligible_qty = eligible_qty_matches.get("content")
        discount_per_qty = f"{price_bundle} par {eligible_qty}"
        return discount_per_qty
    else:
        return None


def _scrap_characteristics(soup: BS) -> str | None:
    """`_scrap_characteristics`: Recupère les caractéristiques principales du vin.
    
    ---------
    `Parameters`
    --------- ::

        soup (BS):

    `Returns`
    --------- ::

        str | None

    `Example(s)`
    --------- ::
    >>> page = requests.get(
        'https://www.vinatis.com/58600-chinon-les-picasses-2017-domaine-olga-raffault')
    >>> soupe = _soupifier(page.text)
    >>> _scrap_characteristics(soupe)
    ... 'Vin Rouge / Loire / Chinon AOC / 12,5 % vol / 100% Cabernet-franc'"""
    characteristics = soup.find(name="span", attrs={"class": "no-padding-horizontal"})

    if isinstance(characteristics, Tag):
        return characteristics.text
    else:
        return None


def _scrap_notes(soup: BS) -> str | None:
    """`_scrap_notes`: Recupère la note et le nombre d'avis.
    
    ---------
    `Parameters`
    --------- ::

        soup (BS): 

    `Returns`
    --------- ::

        str | None

    `Example(s)`
    --------- ::
    >>> page = requests.get(
        'https://www.vinatis.com/58600-chinon-les-picasses-2017-domaine-olga-raffault')
    >>> soupe = _soupifier(page.text)
    >>> _scrap_notes(soupe)
    ... null"""
    notes = soup.find(name="div", attrs={"class": "col-xs-12 padding-bottom-10"})

    if isinstance(notes, Tag):
        if notes:
            return notes.text.strip()

    return None


def _scrap_keywords(soup: BS) -> list[str]:
    """`_scrap_keywords`: Recupère les mots importants 
    mis en avant sur la page du produit.
    
    ---------
    `Parameters`
    --------- ::

        soup (BS): 

    `Returns`
    --------- ::

        str | None

    `Example(s)`
    --------- ::
    >>> page = requests.get(
        'https://www.vinatis.com/58600-chinon-les-picasses-2017-domaine-olga-raffault')
    >>> soupe = _soupifier(page.text)
    >>> _scrap_keywords(soupe)
    ... [
    ...     "Dense",
    ...     "Souple",
    ...     "Griotte"
    ... ]"""
    kwd_class = "margin-right margin-bottom bg-gray-dark taille-md padding-horizontal-30 padding-vertical-5 rounded-corner-3 label"
    matches = soup.find_all(name="span", attrs={"class": kwd_class})
    keywords = [keyword.text for keyword in matches]
    return keywords


def _scrap_other(soup: BS) -> str | None:
    """`_scrap_other`: Récupère d'autres attributs : 
    bio, nouveauté, vigneron indépendant, etc.
    
    ---------
    `Parameters`
    --------- ::

        soup (BS): 

    `Returns`
    --------- ::

        str | None

    `Example(s)`
    --------- ::
    >>> page = requests.get(
        'https://www.vinatis.com/58600-chinon-les-picasses-2017-domaine-olga-raffault')
    >>> soupe = _soupifier(page.text)
    >>> _scrap_other(soupe)
    ... 'Bio'"""
    matches = soup.find_all(
        "div",
        attrs={"class": "margin-top-3 display-inline"},
    )
    if not matches:
        return None

    other_characs = "/".join([other_charac.text.strip() for other_charac in matches])
    return other_characs


def _scrap_img(soup: BS) -> str | list[str] | None:
    """`_scrap_img`: Récupère le lien de l'image de la bouteille
    
    ---------
    `Parameters`
    --------- ::

        soup (BS): 

    `Returns`
    --------- ::

        str | list[str] | None

    `Example(s)`
    --------- ::
    >>> page = requests.get(
        'https://www.vinatis.com/58600-chinon-les-picasses-2017-domaine-olga-raffault')
    >>> soupe = _soupifier(page.text)
    >>> _scrap_img(soupe)
    ... 'https://www.vinatis.com/75407-detail_default/chinon-les-picasses-2017-domaine-olga-raffault.png'"""
    picture = soup.find(
        name="img", attrs={"class": "img-full-width img-max-450 center-block"}
    )
    if isinstance(picture, Tag):
        return picture.get("src")
    else:
        return None


def _scrap_details(soup: BS) -> dict[str, str]:
    """`_scrap_details`: Crée un dictionnaire clé-valeur 
    pour extraire des caractéristiques complémentaires.
    
    ---------
    `Parameters`
    --------- ::

        soup (BS): 

    `Returns`
    --------- ::

        dict[str, str]

    `Example(s)`
    --------- ::
    >>> page = requests.get(
        'https://www.vinatis.com/58600-chinon-les-picasses-2017-domaine-olga-raffault')
    >>> soupe = _soupifier(page.text)
    >>> _scrap_details(soupe)
    ... {'Millésime': '2017',
    ... 'Cépage': '100% Cabernet-franc',
    ... 'Bio': 'Certifié Agriculture Biologique, Certifié Eurofeuille',
    ... 'Goûts': 'Rouge Charnu et fruité',
    ... 'Par Goûts': 'Puissant',
    ... "A l'oeil": 'Robe rouge soutenue',
    ... 'Au nez': 'Arômes de fruits rouges mûrs.',
    ... 'En bouche': 'Bouche structurée, fruitée et aromatique',
    ... 'Température de service': '16-18°C',
    ... 'Service': 'Ouvrir 1h avant le service',
    ... 'Conservation': 'A boire et à garder',
    ... "Jusqu'à": '2030',
    ... 'Accords mets-vin': 'Charcuterie, Viande rouge, Viande blanche, Barbecue',
    ... 'Accords recommandés': 'Viandes goûteuses telles l’agneau ou le bœuf cuisiné. Parfait avec les petits gibiers.'}"""
    key_class = "table-cell-css vertical-align-top padding-vertical-5 nowrap padding-right-10 taille-xs color-gray-darker"
    value_class = "table-cell-css vertical-align-top padding-vertical-5 taille-xs color-gray-darker text-bold"
    keys = soup.find_all("div", attrs={"class": key_class})
    values = soup.find_all("div", attrs={"class": value_class})
    dict_details = dict(
        [(key.text.strip(), value.text.strip()) for key, value in zip(keys, values)]
    )
    return dict_details


def _get_value(dict_details: dict, key: str) -> str | None:
    """`_get_value`: Récupère dans le dictionnaire de détails
    la valeur associée à une clé existante.
    
    ---------
    `Parameters`
    --------- ::

        dict_details: dict
        key: str

    `Returns`
    --------- ::

        str | None

    `Example(s)`
    --------- ::
    >>> page = requests.get(
        'https://www.vinatis.com/58600-chinon-les-picasses-2017-domaine-olga-raffault')
    >>> soupe = _soupifier(page.text)
    >>> dictionnaire = _scrap_details(soupe)
    >>> _get_value(dictionnaire, "Cépage")
    ... '100% Cabernet-franc'"""
    try:
        value = dict_details[key]
    except KeyError:
        value = None
    return value


def scraping(page: str) -> Vin:
    """`scraping`: Agrégateur de tous les fonctions de scraping.
    Le tout est transformé en la data classe Vin
    
    ---------
    `Parameters`
    --------- ::

        page : str

    `Returns`
    --------- ::

        Vin

    `Example(s)`
    --------- ::
    >>> page = requests.get(
        'https://www.vinatis.com/58600-chinon-les-picasses-2017-domaine-olga-raffault')
    >>> scraping(page)
    ... {
    ... "name": "CHINON LES PICASSES 2017 - DOMAINE OLGA RAFFAULT",
    ... "capacity": "0,75 L",
    ... "price": "15,00 €",
    ... "price_bundle": null,
    ... "characteristics": "Vin Rouge / Loire / Chinon AOC / 12,5 % vol / 100% Cabernet-franc",
    ... "note": null,
    ... "keywords": [
    ...     "Dense",
    ...     "Souple",
    ...     "Griotte"
    ... ],
    ... "others": "Bio",
    ... "picture": "https://www.vinatis.com/75407-detail_default/chinon-les-picasses-2017-domaine-olga-raffault.png",
    ... "classification": null,
    ... "millesime": "2017",
    ... "cepage": "100% Cabernet-franc",
    ... "gouts": "Rouge Charnu et fruité",
    ... "par_gouts": "Puissant",
    ... "oeil": "Robe rouge soutenue",
    ... "nez": "Arômes de fruits rouges mûrs.",
    ... "bouche": "Bouche structurée, fruitée et aromatique",
    ... "temperature": "16-18°C",
    ... "service": "Ouvrir 1h avant le service",
    ... "conservation_1": "2030",
    ... "conservation_2": "A boire et à garder",
    ... "accords_vins": "Charcuterie, Viande rouge, Viande blanche, Barbecue",
    ... "accords_reco": "Viandes goûteuses telles l’agneau ou le bœuf cuisiné."}"""
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
    """`create_json`: Crée un fichier semi-structuré JSON
    à partir d'une liste contenant les liens des vins à scraper.
    
    ---------
    `Parameters`
    --------- ::

        all_pages : list[str]

    `Returns`
    --------- ::

        None

    `Example(s)`
    --------- ::
    >>> create_json(all_pages)
    ... Lien n° 1 / 3 scrapé.
    ... Lien n° 2 / 3 scrapé.
    ... Lien n° 3 / 3 scrapé.
    ... "Export en JSON réalisé avec succès dans nom_du_fichier !" """
    root = Path(".").resolve()
    data_folder = root / "data"

    vins = list()
    nb_vins = len(all_pages)
    etape = 1
    for page in all_pages:
        requete = rq.get(page)
        vins.append(scraping(requete.text))
        print(f"Lien n° {etape} / {nb_vins} scrapé.")
        etape = etape + 1

    vins_json = to_json(vins)
    file_path = data_folder / "vins2.json"

    with open(file_path, "w", encoding="utf-8") as json_file:
        json_file.write(vins_json)

    return print(f"Export en JSON réalisé avec succès dans {data_folder} !")
