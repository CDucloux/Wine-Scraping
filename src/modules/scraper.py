from modules.scraping_functions import *
from pathlib import Path

root = Path(".").resolve()
data_folder = root / "data"

url_browse_list = create_urls_browse_list()
session = create_session()
all_wine_urls = create_all_wine_urls(session, url_browse_list)

# TODO : créer une fonction pour l'écriture des liens vers un csv.
with open(data_folder / "wine_links.csv", "w") as file_path:
    file_path.write(";\n".join(all_wine_urls))
