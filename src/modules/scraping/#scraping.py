"""MARCHE"""
import subprocess

subprocess.call(['python', "src\modules\scraping\scraper.py"])
subprocess.call(['python', "src\modules\scraping\pages_scraping_script.py"])