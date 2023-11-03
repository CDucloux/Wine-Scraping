"""
module permettant de réaliser le webscraping avec selenium & bs4

Exemple de fonctionnement : se.build_wine("https://www.vinatis.com/rhum-ron-rum")
Et.. c'est fini.
"""
import time
import re
import requests as rq
from bs4 import BeautifulSoup as BS
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium import webdriver
import modules.old.data_extractor as de
from src.modules.vin_dataclass import Vin


def _initialise_chrome(lien: str):
    """
    Lance chrome, ferme les cookies et les pop-up

    Args:
        lien (str): lien des pages à scrapper

    Returns:
        _type_: driver
    """
    service = Service(executable_path="./chromedriver.exe")
    driver = webdriver.Chrome(service=service)
    driver.get(lien)
    cookie_accept_button = driver.find_element(
        By.XPATH, "//button[contains(text(), 'Accepter et continuer')]"
    )
    cookie_accept_button.click()

    time.sleep(2)
    driver.execute_script("window.scrollBy(0, 500);")
    while True:
        try:
            time.sleep(10)
            driver.find_element(By.CLASS_NAME, "close_modal_vinatis").click()
        except:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            driver.execute_script("window.scrollTo(0, 0);")
            continue
        break

    return driver


def _nb_pages(driver: webdriver.Chrome) -> int():
    """_summary_

    Args:
        driver (_type_):

    Returns:
        int : le nombre de pages à scraper
    """
    page_source = driver.page_source
    soupe = BS(page_source)
    result_nb_pages = soupe.find_all(
        name="span", attrs={"class": "color-light-grey margin-right-10"}
    )
    nb_pages = int(re.findall(r"\d+", result_nb_pages[0].text)[0])
    return nb_pages


def _change_page(driver):
    """
    Change de page en laissant du temps entre chaque opération pour éviter tout bugs.

    Args:
        driver (_type_):
    """
    time.sleep(2)
    bouton = driver.find_element(By.CLASS_NAME, "icon-chevron-right")
    time.sleep(2)
    driver.execute_script("arguments[0].scrollIntoView();", bouton)
    time.sleep(2)
    driver.execute_script("window.scrollBy(0, -500);")
    time.sleep(2)
    bouton.click()
    time.sleep(2)
    return None


def _scraping_produits_page(driver, vins: list):
    """
    Récupère le contenu d'une page puis parcours chaque
    produit de la page pour récuperer des données

    Args:
        driver (_type_):
        vins (list):

    Returns:
        list : liste des vins mise à jours
    """
    page_source = driver.page_source
    soupe = BS(page_source)

    for item_1, item_2, item_3, item_4, item_5 in zip(
        de.extract_result(soupe)["result_title"],
        de.extract_result(soupe)["result_price"],
        de.extract_result(soupe)["result_avis"],
        de.extract_result(soupe)["result_type"],
        de.extract_result(soupe)["result_title_notxt"],
    ):
        vins.append(de.extract_vin(item_1, item_2, item_3, item_4, item_5))

    return vins


def scrapeur(lien: str):
    """
    Agregation des fonctions permettant de réaliser le webscraping

    Args:
        lien (str): page(s) à scraper

    Returns:
        list(): liste des produits scrapés
    """
    driver = _initialise_chrome(lien)
    nombre_page = _nb_pages(driver)
    i = 1
    vins = list()
    while i < nombre_page:
        _scraping_produits_page(driver, vins)
        _change_page(driver)
        i = i + 1

    driver.quit()

    return vins


def _infos(text: str):
    """
    expression régulière pour récupérer le degrés d'alcool (url_vol)
    """
    match = re.findall(r"[^/]+", text)
    return match


def _url_vol(soupe: BS, vol: list()):
    """
    Permet de récupérer la donnée "vol"
    Eventuellement le "type"
    """
    try:
        text_carac = soupe.find_all(
            name="span",
            attrs={
                "class": "ss-titre color-gray-darker taille-xs line-height-15-xs no-padding-horizontal"
            },
        )[0].text
        # type.append(infos(text_carac)[0])

        if _infos(text_carac)[-1].endswith("vol"):
            vol.append(_infos(text_carac)[-1])
        else:
            vol.append(_infos(text_carac)[-2])
    except:
        vol.append(None)

    return vol


def _url_adjectifs(soupe: BS, adjectif: list()):
    """
    Permet de récupérer les données "adjectifs"
    """
    try:
        adjectif.append(
            soupe.find_all(name="div", attrs={"class": "margin-top-5 hide-xs"})[0].text
        )
    except:
        adjectif.append(None)

    return adjectif


def _url_cepages(soupe: BS, cepage: list()):
    """
    Permet de récupérer les données de "cépages"
    """
    try:
        text_cepage = soupe.find_all(
            name="div",
            attrs={
                "class": "table-cell-css vertical-align-top padding-vertical-5 taille-xs color-gray-darker text-bold"
            },
        )[1].text
        cepage.append(text_cepage)
    except:
        cepage.append(None)

    return cepage


def _extract_infos_url(vins: Vin):
    """
    Compile les fonctions récupérants les données
    """
    adjectif = list()
    vol = list()
    cepage = list()

    for vin in vins:
        requete = rq.get(url=vin.lien, timeout=10)
        soupe = BS(requete.text)
        vol = _url_vol(soupe, vol)
        adjectif = _url_adjectifs(soupe, adjectif)
        cepage = _url_cepages(soupe, cepage)

    return {"adjectif": adjectif, "vol": vol, "cepage": cepage}


def actualisation_vin(vins: Vin):
    """
    Met à jours la base de données avec les informations récuperer sur la pages des produits
    """
    extraction = _extract_infos_url(vins)
    vols = extraction["vol"]
    adjectifs = extraction["adjectif"]
    cepages = extraction["cepage"]

    for vin, vol, adjectif, cepage in zip(vins, vols, adjectifs, cepages):
        vin.vol = vol
        vin.adjective = adjectif
        vin.cepage = cepage

    return vins


def build_wine(lien):
    """
    Input le lien de la page à scraper
    Output database des pages
    """
    return actualisation_vin(scrapeur(lien))
