from src.modules.scraping.mystical_soup import (
    _soupifier,
    _scrap_name,
    _scrap_capacity,
    _scrap_price,
    _scrap_price_bundle,
    _scrap_characteristics,
    _scrap_notes,
    _scrap_keywords,
    _scrap_other,
    _scrap_img,
    _scrap_details
)
from bs4 import BeautifulSoup as BS

with open(r"tests\files\page.txt", 'r') as fichier:
    REQUETE = fichier.read()
    
PAGE = _soupifier(REQUETE)

def test__soupifier():
    assert isinstance(_soupifier(REQUETE), BS)

def test__scrap_name():
    assert isinstance(_scrap_name(PAGE), str)
    assert _scrap_name(PAGE) == "LES DARONS 2022 - BY JEFF CARREL"

def test__scrap_capacity():
    assert _scrap_capacity(PAGE) == "0,75 L"

def test__scrap_characteristics():
    assert _scrap_characteristics(PAGE) == "Vin Rouge / Languedoc-Roussillon / Languedoc AOC / 14 % vol"

def test__scrap_notes():
    assert _scrap_notes(PAGE) == "16.8/20  noté par 46 clients"
    
def test__scrap_other():
    assert _scrap_other(PAGE) == r"Coup de c\\xc5\\x93ur Clients/VINS EN F\\xc3\\x8aTES"

def test__scrap_details():
    assert isinstance(_scrap_details(PAGE), dict)

def test__scrap_price():
    assert _scrap_price(PAGE) == r"9,10 \\xe2\\x82\\xac   6,50 \\xe2\\x82\\xac    -28%"

def test__scrap_keywords():
    assert _scrap_keywords(PAGE) == ['Gourmand', 'Meilleure Vente', 'N\\\\xc2\\\\xb01 Languedoc-Roussillon']
    
def test__scrap_img():
    assert _scrap_img(PAGE) == "https://www.vinatis.com/78830-detail_default/les-darons-2022-by-jeff-carrel.png"

def test__scrap_price_bundle():
    assert _scrap_price_bundle(PAGE) is None