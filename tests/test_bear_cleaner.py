from src.modules.bear_cleaner import (
    _get_avg_temp,
    _get_valid_millesime,
    _get_conservation_date,
    _get_unit_and_offer_price,
    _get_keywords,
    _get_capacity,
    _get_bio,
    _get_customer_fav,
    _get_new,
    _get_top_100,
    _get_destock,
    _get_sulphite_free,
    _get_alcohol_volume,
    _get_type,
    _get_country,
    _get_iso_country_code,
    _get_bubbles,
    _get_new_type,
    _get_cepage,
    _get_wine_note,
    _get_reviews,
    _get_service,
    _get_conservation_time,
    _is_cru,
    _drop_price,
    super_pipe
)
import polars as pl

df_brut = pl.read_json(r"tests\files\test_data.json")[2]

def test__get_avg_temp():
    df = _get_avg_temp(df_brut)
    assert df.select("avg_temp").item() == 7.0
    assert df.select("temp_low").item() == 6.0
    assert df.select("temp_high").item() == 8.0

def test__get_valid_millesime():
    df = _get_valid_millesime(df_brut)
    assert df.select("millesime").item() == "2022"

def test__get_conservation_date():
    df = _get_conservation_date(df_brut)
    assert df.select("conservation_date").item() == 2027

def test__get_unit_and_offer_price():
    df = _get_unit_and_offer_price(df_brut)
    assert df.select("unit_price").item() == 11.7

def test__get_keywords():
    df = _get_keywords(df_brut)
    assert df.select("keyword_1").item() == "Fraîcheur"
    assert df.select("keyword_2").item() == "100% Chenin"
    assert df.select("keyword_3").item() == "Équilibre"

def test__get_capacity():
    df = _get_capacity(df_brut)
    assert df.select("capacity").item() == 0.75

def test__get_bio():
    df = _get_bio(df_brut)
    assert df.select("bio").item() == 0

def test__get_customer_fav():
    df = _get_customer_fav(df_brut)
    assert df.select("customer_fav").item() == 1

def test__get_new():
    df = _get_new(df_brut)
    assert df.select("is_new").item() == 0

def test__get_top_100():
    df = _get_top_100(df_brut)
    assert df.select("top_100").item() == 0

def test__get_destock():
    df = _get_destock(df_brut)
    assert df.select("destock").item() == 0

def test__get_sulphite_free():
    df = _get_sulphite_free(df_brut)
    assert df.select("sulphite_free").item() == 0

def test__get_alcohol_volume():
    df = _get_alcohol_volume(df_brut)
    assert df.select("alcohol_volume").item() == 13.0

def test__get_type():
    df = _get_type(df_brut)
    assert df.select("type").item() == "Vin Blanc"

def test__get_country():
    df = _get_country(df_brut)
    assert df.select("country").item() == "France"

def test__get_iso_country_code():
    df = _get_country(df_brut)
    df = _get_iso_country_code(df)
    assert df.select("iso_code").item() == "FRA"

def test__get_bubbles():
    df = _get_type(df_brut)
    df = _get_bubbles(df)
    assert df.select("bubbles").item() == 0

def test__get_new_type():
    df = _get_type(df_brut)
    df = _get_bubbles(df)
    df = _get_new_type(df)
    assert df.select("type").item() == "Vin Blanc"

def test__get_cepage():
    df = _get_cepage(df_brut)
    assert df.select("cepage").item() == "Chenin"

def test__get_wine_note():
    df = _get_wine_note(df_brut)
    assert df.select("wine_note").item() == 4.3

def test__get_reviews():
    df = _get_reviews(df_brut)
    assert df.select("nb_reviews").item() == 10

def test__get_service():
    df = _get_service(df_brut)
    assert df.select("service").item() == "En bouteille"

def test__get_conservation_time():
    df = _get_valid_millesime(df_brut)
    df = _get_conservation_date(df)
    df = _get_conservation_time(df)
    assert df.select("conservation_time").item() == 5

def test__is_cru():
    df = _is_cru(df_brut)
    assert df.select("cru").item() == 0

def test__drop_price():
    df = _get_unit_and_offer_price(df_brut)
    df = _drop_price(df)
    assert df.select("unit_price").item() == 11.7

def test_super_pipe():
    df = super_pipe(df_brut)
    assert df.shape == (1,39)