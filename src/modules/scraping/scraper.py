from scraping_functions import *
from pathlib import Path

root = Path(".").resolve()
data_folder = root / "data"

url_browse_list = create_urls_browse_list()
session = create_session()
all_wine_urls = create_all_wine_urls(session, url_browse_list)

export_wine_links(data_folder, all_wine_urls)
