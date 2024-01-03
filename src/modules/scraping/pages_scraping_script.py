"""Module permettant de crée un JSON de tout les vins grâce aux liens récupéré avec scraper.py"""
from mystical_soup import create_json
import pandas as pd

df = pd.read_csv("./data/wine_links.csv")
wine_links = list(df.iloc[:, 0])
    
create_json(wine_links)