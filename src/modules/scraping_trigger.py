"""Exécute les scripts qui permettent de récupérer les liens des pages individuelles des vins 
avant de les scrapper une à une et des les exporter en JSON :

- page_scraper
- wine_scraper
"""
import subprocess

subprocess.call(["python", "src.modules.scraping.page_scraper.py"])
subprocess.call(["python", "src.modules.scraping.wine_scraper.py"])
