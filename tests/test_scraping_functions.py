from src.modules.scraping.scraping_functions import (
    random_waiter,
    get_random_proxy,
    get_random_user_agent,
    create_session,
    random_true_false,
    update_session,
    create_urls_browse_list,
    _catch_url,
    export_wine_links,
    extract_all_pages,
    timer
)
from requests_html import HTMLSession 
from pathlib import Path
from typing import Callable

def test_get_random_proxy():
    assert isinstance(get_random_proxy(),str)

def test_get_random_user_agent():
    rua = get_random_user_agent()
    assert isinstance(rua, str)
    assert rua in [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9",
        "Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36",
    ]
    
def test_create_session():
    assert isinstance(create_session(), HTMLSession)

def test_random_true_false():
    assert random_true_false(prob_false=1) == False
    assert random_true_false(prob_false=0) == True

def test_update_session():
    session = create_session()
    assert isinstance(update_session(session), HTMLSession)

def test_create_urls_browse_list():
    assert isinstance(create_urls_browse_list(),list)

def test_export_wine_links():
    root = Path(".").resolve()
    data_folder = root / "tests/files"
    export_wine_links(data_folder, ["lien.com", "unautre.com"])
    assert (data_folder / "wine_links.csv").is_file() == True
    
def test_extract_all_pages():
    session = create_session()
    eap = extract_all_pages(session, ["https://www.vinatis.com/60679-chateau-moulin-bellegrave-2020"])
    assert str(eap[0][0:10]) == str("\n<!DOCTYPE")

def test__catch_url():
    session = create_session()
    page = _catch_url(session, "https://www.vinatis.com/60679-chateau-moulin-bellegrave-2020", 1, render=False)
    assert str(page[0:10]) == str("\n<!DOCTYPE")

def test_random_waiter():
    assert isinstance(random_waiter(0.1,0.2), Callable)

def test_timer():
    assert isinstance(timer(get_random_proxy()), Callable)