"""
`scraping_functions` : à vos verres 🍷 !

ce module comporte l'ensemble des fonctions de scraping nécessaires pour extraire les vins de nos terroirs.
"""

from yarl import URL
from requests_html import HTMLSession
from bs4 import BeautifulSoup
from itertools import chain
from http.client import RemoteDisconnected
import re
import time
from rich import print as rprint

URL_INIT = URL.build(scheme="https", host="vinatis.com")
WHITE = "achat-vin-blanc"
RED = "achat-vin-rouge"
ROSE = "achat-vin-rose"


def timer(func):
    """`timer`: This function  allows to be passed as a @decorator and gives the elapsed time of a function afterwards.

    ---------
    `Parameters`
    --------- ::

        func (function): # Any function basically

    `Example(s)`
    ---------

    >>> @timer
    >>> def func(a:int)-> list[int]:
    >>>     return [a for a in range(1000000)]
    >>> func(2)
    ... #Elapsed time for func function: 0.041 seconds."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        rprint(
            f"""
--------------------------------\n
[italic]Elapsed time[/italic] for [bold red]{func.__name__}[/bold red] function: 
{round(elapsed_time,3)} seconds.
"""
        )
        return result

    return wrapper


def create_session() -> HTMLSession:
    """`create_session`: Crée une session HTML avec un user-agent spécifique.

    - Fait croire au navigateur qu'un utilisateur envoie une requête, et non un robot.

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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0"
        }
    )
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
            page_range = range(1, 5)  # (1,50)
        elif suffix is RED:
            page_range = range(1, 5)  # (1,97)
        elif suffix is ROSE:
            page_range = range(1, 5)  # (1,9)
        for page in page_range:
            url_browse_list.append(URL_INIT / suffix % {"page": page, "tri": 7})
    return url_browse_list


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
                r.html.render()
                page = r.html.html
            else:
                page = r.html.html
            return page
        else:
            print(f"Page {page_number+1} -> Unsuccessfull Extraction.")
            pass
    except RemoteDisconnected:
        print(f"Page {page_number+1} -> Server Connection Error.")
        pass


def _href_finder(page: str, URL_INIT=URL_INIT) -> list[str]:
    """`_href_finder`: Trouve tous les hrefs valides dans une page de recherche de vins."""
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

    - Grâce à la liste d'URL des pages de recherche, cette fonction va chercher les href et les complète.

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
        page = _catch_url(session, url, page_number, render=True)
        valid_hrefs = _href_finder(page)
        all_wine_links.append(valid_hrefs)
    all_wine_links = list(chain.from_iterable(all_wine_links))
    return all_wine_links


@timer
def extract_all_pages(session: HTMLSession, all_wine_links: list[str]) -> list[str]:
    """`extract_all_pages`:

    - Extrait le contenu brut de toutes les pages web de vin sans rendu javascript.

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
    return wine_pages
