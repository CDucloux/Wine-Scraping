"""
`scraping_functions` : à vos verres 🍷 !

Ce module comporte l'ensemble des fonctions de scraping
nécessaires pour extraire les vins de nos terroirs.
"""

from yarl import URL
from requests_html import HTMLSession  # type: ignore
from bs4 import BeautifulSoup
from itertools import chain
from fp.fp import FreeProxy  # type: ignore
from http.client import RemoteDisconnected
from typing import Callable
from pathlib import Path
import re
import time
import random
from rich import print as rprint

URL_INIT = URL.build(scheme="https", host="vinatis.com")
WHITE = "achat-vin-blanc"
RED = "achat-vin-rouge"
ROSE = "achat-vin-rose"


def random_waiter(min_wait: float, max_wait: float) -> Callable:
    """`random_waiter`: @decorator -> Renvoie un temps d'attente aléatoire.

    ---------
    `Parameters`
    --------- ::

        min_wait (float): # Temps minimum
        max_wait (float): # Temps maximum

    `Returns`
    --------- ::

        Callable

    `Example(s)`
    ---------

    >>> @random_waiter
    >>> def func(a:int)-> None:
    >>>     return None
    >>> func(2)
    ... --------------------------------
    ... Random Timer:
    ... 0.509 seconds.
    """

    def decorator(func) -> Callable:
        def wrapper(*args, **kwargs):
            wait_time = random.uniform(min_wait, max_wait)
            time.sleep(wait_time)
            rprint(
                f"""
--------------------------------\n
[italic]Random Timer[/italic]: 
{round(wait_time, 3)} seconds.
"""
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def timer(func) -> Callable:
    """`timer`: @decorator -> Renvoie le temps d'éxécution d'une fonction.

    ---------
    `Parameters`
    --------- ::

        func (function): # N'importe quelle fonction

    `Example(s)`
    ---------

    >>> @timer
    >>> def func(a:int)-> list[int]:
    >>>     return [a for a in range(1000000)]
    >>> func(2)
    ... Elapsed time for func function: 0.041 seconds."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        rprint(
            f"""
--------------------------------\n
[italic]Elapsed time[/italic] for [bold red]{func.__name__}[/bold red] function: 
{round(elapsed_time, 3)} seconds.
"""
        )
        return result

    return wrapper


def get_random_proxy() -> str:
    """`get_random_proxy`: Génère un proxy aléatoire.

    `Returns`
    --------- ::

        str

    `Example(s)`
    ---------

    >>> get_random_proxy()
    ... "http://178.128.160.79:80"
    """
    proxy = FreeProxy(country_id=["GB"], rand=True).get()
    return proxy


def get_random_user_agent() -> str:
    """`get_random_user_agent`: Renvoie un User-Agent aléatoire.

    `Returns`
    --------- ::

        str

    `Example(s)`
    ---------

    >>> get_random_user_agent()
    ... "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0"
    """
    valid_user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9",
        "Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36",
    ]
    user_agent = random.choice(valid_user_agents)
    return user_agent


def random_true_false(prob_false=0.85):
    weights = [1 - prob_false, prob_false]
    return random.choices([True, False], weights=weights)[0]


def create_session() -> HTMLSession:
    """`create_session`: Crée une session HTML avec un proxy
    et un user-agent spécifique aléatoire.

    - Fait croire au navigateur qu'un utilisateur envoie une
    requête, et non un robot.
    - Cache l'adresse IP du client grâce à un proxy.

    `Returns`
    --------- ::

        HTMLSession

    `Example(s)`
    ---------

    >>> create_session()
    ... <requests_html.HTMLSession at 0x1e0c6506020>"""
    session = HTMLSession()
    session.headers.update(
        {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3",
            "User-Agent": get_random_user_agent(),
        }
    )
    session.proxies.update({"http": get_random_proxy()})
    return session


def update_session(session: HTMLSession) -> HTMLSession:
    """`update_session`:

    - Met à jour la session dans 15% des cas avec un nouveau
    proxy et un User-Agent aléatoire.

    ---------
    `Parameters`
    --------- ::

        session (HTMLSession): # session HTML

    `Returns`
    --------- ::

        HTMLSession

    `Example(s)`
    ---------

    >>> update_session()
    ... #TODO"""
    update = random_true_false()
    print(f"Session updated = {update}")
    if update is True:
        session = create_session()
    else:
        session = session
    return session


def create_urls_browse_list(URL_INIT=URL_INIT) -> list[URL]:
    """`create_urls_browse_list`: Crée la liste des URL de page de recherche de vins :

    - VIN ROUGE
    - VIN BLANC
    - ROSE

    ---------
    `Parameters`
    --------- ::

        URL_INIT (URL, optional): # Defaults to URL_INIT.

    `Returns`
    --------- ::

        list[URL]

    `Example(s)`
    ---------

    >>> create_urls_browse_list()
    ... [
    ... URL('https://vinatis.com/achat-vin-rouge?page=1&tri=7'),
    ... URL('https://vinatis.com/achat-vin-rouge?page=2&tri=7')
    ... ]
    """
    url_browse_list = list()
    for suffix in {WHITE, RED, ROSE}:
        if suffix is WHITE:
            page_range = range(1, 50)  # (1,50)
        elif suffix is RED:
            page_range = range(1, 97)  # (1,97)
        elif suffix is ROSE:
            page_range = range(1, 9)  # (1,9)
        for page in page_range:
            url_browse_list.append(URL_INIT / suffix % {"page": page, "tri": 7})
    random.shuffle(url_browse_list)
    return url_browse_list


@random_waiter(0.25, 0.75)
@timer
def _catch_url(
    session: HTMLSession, url: URL, page_number: int, render: bool
) -> str | None:
    """`_catch_url`: envoie une requête et obtient si possible le code source HTML de la page.

    - Si l'argument `render` est True, alors le javascript sera éxécuté et WYSIWYG.
    - Gère aussi les problèmes de déconnexion de la part du serveur.
    """
    try:
        r = session.get(url)
        if r.status_code == 200:
            print(f"Page {page_number+1} -> Successfull Extraction.")
            if render:
                r.html.render(timeout=25)
                page = r.html.html
            else:
                page = r.html.html
        else:
            print(f"Page {page_number+1} -> Unsuccessfull Extraction.")
            page = None
    except RemoteDisconnected:
        print(f"Page {page_number+1} -> Server Connection Error.")
        page = None
    return page


def _href_finder(page: str, URL_INIT=URL_INIT) -> list[str]:
    """`_href_finder`: Trouve tous les hrefs valides dans une
    page de recherche de vins."""
    soup = BeautifulSoup(page, "html.parser")
    a_tags = soup.select("a.display-block")
    hrefs = [a.get("href") for a in a_tags]
    valid_hrefs = [
        str(URL_INIT / href[1:]) for href in hrefs if re.match(r"^\d+", href[1:])
    ]
    return valid_hrefs


@timer
def create_all_wine_urls(session: HTMLSession, url_browse_list: list[URL]) -> list[str]:
    """`create_all_wine_urls`:

    - Grâce à la liste d'URL des pages de recherche, 
    cette fonction va chercher les href et les complète.
    - Elle retire aussi les doublons potentiels grâce à 
    la transformation `list -> set -> list`

    ---------
    `Parameters`
    --------- ::

        session (HTMLSession): # session HTML
        url_browse_list (list[URL]): # URL des pages de recherche

    `Returns`
    --------- ::

        list[str] # liste d'URLs de tous les vins

    `Example(s)`
    ---------

    >>> create_all_wine_urls()
    ... #TODO"""
    all_wine_links = list()
    for page_number, url in enumerate(url_browse_list):
        session = update_session(session)
        page = _catch_url(session, url, page_number, render=True)
        valid_hrefs = _href_finder(page)
        all_wine_links.append(valid_hrefs)
    all_wine_links = list(chain.from_iterable(all_wine_links))
    return list(set(all_wine_links))


def export_wine_links(folder_path: Path, all_wine_links: list[str]) -> None:
    """`export_wine_links`:

    - Exporte les liens de l'ensemble des pages de vins vers un fichier csv.

    ---------
    `Parameters`
    --------- ::

        path_folder (Path): # Chemin du dossier où enregistrer le csv
        all_wine_links (list[str]): # liste des liens des vins

    `Returns`
    --------- ::

        None

    `Example(s)`
    ---------

    >>> export_wine_links()
    ... Export réalisé dans D:\Cours Mecen (M2)\Machine Learning\Wine Scraping\data.
    """
    with open(folder_path / "wine_links.csv", "w") as file_path:
        file_path.write(";\n".join(all_wine_links))
    return print(f"Export réalisé dans {folder_path}.")


@timer
def extract_all_pages(session: HTMLSession, all_wine_links: list[str]) -> list[str]:
    """`extract_all_pages`:

    - Extrait le contenu brut de toutes les pages web de vin sans rendu javascript.
    - Retire les pages dont la requête n'a rien renvoyé.

    ---------
    `Parameters`
    --------- ::

        session (HTMLSession): # session HTML
        all_wine_links (list[str]): # URL des pages des vins

    `Returns`
    --------- ::

        list[str]

    `Example(s)`
    ---------

    >>> extract_all_pages()
    ... #TODO"""
    wine_pages = list()
    for page_number, url in enumerate(all_wine_links):
        content_page = _catch_url(session, url, page_number, render=False)
        wine_pages.append(content_page)
    return list(filter(None, wine_pages))
