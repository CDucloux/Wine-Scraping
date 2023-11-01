import requests as rq
import re
from bs4 import BeautifulSoup as BS
from serde.json import to_json
from src.modules.scraping_class import Vin

def _scrap_soupe(adresse:str) -> BS:
    requete = rq.get(url=adresse, timeout=10)
    soupe = BS(requete.text)
    return soupe

def _scrap_name(soupe:BS) -> str:
    name = soupe.find_all(name="span", attrs={"itemprop": "name"})[0].text
    return name

def _scrap_capacity(soupe:BS) -> str:
    capacity = soupe.find_all(name="span", attrs={"inline-block"})[0].text
    return capacity

def _scrap_price(soupe:BS) -> str:
    price = soupe.find_all(name="span", attrs={"id" : "our_price_display"})[0].text
    
    try:
        price_discount_1 = soupe.find_all(name="span", attrs={"id" : "quantity_discount_pretaxe"})[0].text
    except:
        price_discount_1 = None
        
    try:
        price_discount_2 = soupe.find_all(name="span", attrs={"id" : "quantity_discount_pretaxe"})[1].text
    except:
        price_discount_2 = None
    return price, price_discount_1, price_discount_2

def _scrap_characteristic(soupe:BS)->str:
    characteristic = soupe.find_all(name="span", attrs={"class": "no-padding-horizontal"})[0].text
    return characteristic

def _scrap_note(soupe:BS)-> str:
    note = soupe.find_all(name="div", attrs={"class": "col-xs-12 padding-bottom-10"})[0].text
    return note

def _scrap_keyword(soupe:BS)-> list:
    keyword = []
    class_k = "margin-right margin-bottom bg-gray-dark taille-md padding-horizontal-30 padding-vertical-5 rounded-corner-3 label"
    result = soupe.find_all(name = "span", attrs={"class": class_k})
    for res in result:
        keyword.append(res.text)
    return keyword

def _scrap_informations(soupe:BS) -> str:
    class_ = "table-cell-css vertical-align-top padding-vertical-5 taille-xs color-gray-darker text-bold"
    vintage = soupe.find_all(name="div", attrs={"class": class_})[0].text
    grap_variety = soupe.find_all(name="div", attrs={"class": class_})[1].text
    
    taste = soupe.find_all(name="div", attrs={"class": class_})[2].text
    taste_group = soupe.find_all(name="div", attrs={"class": class_})[3].text
    eyes = soupe.find_all(name="div", attrs={"class": class_})[4].text
    nose = soupe.find_all(name="div", attrs={"class": class_})[5].text
    mouth = soupe.find_all(name="div", attrs={"class": class_})[6].text
    
    temperature = soupe.find_all(name="div", attrs={"class": class_})[7].text
    indication = soupe.find_all(name="div", attrs={"class": class_})[8].text
    conservation = soupe.find_all(name="div", attrs={"class": class_})[9].text
    max_date = soupe.find_all(name="div", attrs={"class": class_})[10].text
    
    accompaniement = soupe.find_all(name="div", attrs={"class": class_})[11].text
    
    return vintage, grap_variety, taste, taste_group, eyes, nose, mouth, temperature, indication, conservation, max_date, accompaniement

def _scrap_img(soupe:BS) -> str:
    picture = soupe.find(name = "img", attrs={"class":"img-full-width img-max-450 center-block"}).get('src')
    return picture

def _scraping(adresse):
    soupe = _scrap_soupe(adresse)
    
    results = Vin(
        _scrap_name(soupe),
        _scrap_capacity(soupe),
        _scrap_price(soupe),
        _scrap_characteristic(soupe),
        _scrap_note(soupe),
        _scrap_keyword(soupe),
        _scrap_informations(soupe),
        _scrap_img(soupe)
        )
    
    return results

def json_brut(adresse):
    vins = _scraping(adresse)
    vins_json = to_json(vins)
    
    file_path = "data/vins.json"

    with open(file_path, "w",  encoding='utf-8') as json_file:
        json_file.write(vins_json)