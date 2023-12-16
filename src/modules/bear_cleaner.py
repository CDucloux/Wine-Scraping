"""
`bear_cleaner`
==============

L'ours polaire qui nettoie la donnée 🐻
"""
import polars as pl

# TODO: implémentation LazyFrame
# TODO 2: respecter l'interface privée des fonctions, utiliser doctest et mypy pour documenter.


def get_avg_temp(df: pl.DataFrame) -> pl.DataFrame:
    """Obtient la température moyenne du vin à partir d'une valeur au format <10-12°C>."""
    df = (
        df.with_columns(
            pl.col("temperature")
            .str.replace("°C", "")
            .str.split("-")
            .list.to_struct()
            .struct.rename_fields(["temp_low", "temp_high"])
        )
        .unnest("temperature")
        .with_columns(pl.col("temp_low", "temp_high").cast(pl.Float64))
        .with_columns(
            pl.coalesce(pl.col("temp_high"), pl.col("temp_low")).alias("temp_high")
        )
        .with_columns(
            pl.col("temp_low").add(pl.col("temp_high")).truediv(2).alias("avg_temp")
        )
    )
    return df


def get_valid_millesime(df: pl.DataFrame) -> pl.DataFrame:
    """Transforme les valeurs aberrantes de la colonne millésime."""
    df = df.with_columns(
        pl.when(pl.col("millesime") == "non millésimé")
        .then(None)
        .otherwise(pl.col("millesime"))
        .alias("millesime")
    )
    return df


def get_conservation_date(df: pl.DataFrame) -> pl.DataFrame:
    """Obtient la date de conservation du vin."""
    df = (
        df.with_columns(
            pl.when(pl.col("conservation_2") == "A boire dans les 2 ans")
            .then(pl.lit("2"))
            .otherwise(pl.lit("0"))
            .alias("conservation_2")
            .cast(pl.Int64)
        )
        .with_columns(pl.col("conservation_1", "millesime").cast(pl.Int64))
        .with_columns(
            pl.coalesce(pl.col("conservation_1"), pl.col("millesime")).alias(
                "conservation_1"
            )
        )
        .with_columns(
            pl.col("conservation_1")
            .add(pl.col("conservation_2"))
            .alias("conservation_date")
        )
    )
    df = df.drop("conservation_1", "conservation_2")
    return df


def get_unit_and_offer_price(df: pl.DataFrame) -> pl.DataFrame:
    """Extrait le prix unitaire et le prix promotionnel."""
    df = (
        df.with_columns(
            pl.col("price")
            .str.extract_all(r"\d+,\d+ €")
            .list.to_struct("max_width")  # stratégie d'expansion maximum de la liste
            .struct.rename_fields(["unit_price", "offer_price"])
        )
        .unnest("price")
        .with_columns(
            pl.col("unit_price", "offer_price")
            .str.replace(",", ".")
            .str.replace(" €", "")
            .cast(pl.Float64)
        )
    )
    return df


def get_keywords(df: pl.DataFrame) -> pl.DataFrame:
    """Obtient les mots-clés associés au vin et conserve la colonne initiale de keywords."""
    df = (
        df.with_columns(pl.col("keywords").alias("keywords_2"))
        .with_columns(
            pl.col("keywords_2")
            .list.to_struct()
            .struct.rename_fields(["keyword_1", "keyword_2", "keyword_3"])
        )
        .unnest("keywords_2")
    )
    return df


def get_capacity(df: pl.DataFrame) -> pl.DataFrame:
    """Obtient la capacité en litres du vin."""
    df = df.with_columns(
        pl.col("capacity")
        .str.split_exact("L", 1)
        .struct[0]
        .str.strip_chars()
        .str.replace(",", ".")
        .alias("capacity")
        .cast(pl.Float64)
    )
    return df


def get_bio(df: pl.DataFrame) -> pl.DataFrame:
    """Indique si le vin est bio."""
    df = df.with_columns(
        pl.when(pl.col("others").str.contains("Bio"))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("bio")
    )
    return df


def get_customer_fav(df: pl.DataFrame) -> pl.DataFrame:
    """Indique si le vin est un coup de coeur client."""
    df = df.with_columns(
        pl.when(pl.col("others").str.contains("Coup de cœur Clients"))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("customer_fav")
    )
    return df


def get_new(df: pl.DataFrame) -> pl.DataFrame:
    """Indique si le vin est une nouveauté sur le site."""
    df = df.with_columns(
        pl.when(pl.col("others").str.contains("Nouveauté"))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("is_new")
    )
    return df


def get_top_100(df: pl.DataFrame) -> pl.DataFrame:
    """Indique si le vin fait partie d'un TOP 100."""
    df = df.with_columns(
        pl.when(pl.col("others").str.to_uppercase().str.contains("TOP 100"))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("top_100")
    )
    return df


def get_destock(df: pl.DataFrame) -> pl.DataFrame:
    """Indique si le vin fait partie d'une opération de déstockage."""
    df = df.with_columns(
        pl.when(pl.col("others").str.contains("Destockage"))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("destock")
    )
    return df


def get_sulphite_free(df: pl.DataFrame) -> pl.DataFrame:
    """Indique si le vin est sans sulfites."""
    df = df.with_columns(
        pl.when(pl.col("others").str.contains("SANS SULFITE"))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("sulphite_free")
    )
    df = df.drop("others")
    return df


def get_alcohol_volume(df: pl.DataFrame) -> pl.DataFrame:
    """Obtient le degré d'alcool."""
    df = df.with_columns(
        pl.col("characteristics")
        .str.extract(r"\b(\d+(?:,\d+)?\s*% vol)\b")
        .str.replace(" % vol", "")
        .str.replace(",", ".")
        .alias("alcohol_volume")
        .cast(pl.Float64)
    )
    return df


def get_type(df: pl.DataFrame) -> pl.DataFrame:
    """Obtient le type de vin : blanc/rouge/rosé."""

    df = df.with_columns(
        pl.col("characteristics")
        .str.split_exact("/", 1)
        .struct[0]
        .str.strip_chars()
        .str.replace("Cubi", "Vin")
        .str.replace("Pack \(carton panaché\)", "Vin")
        .alias("type")
    )
    return df


def get_country(df: pl.DataFrame) -> pl.DataFrame:
    """Permet d'extraire le pays."""
    REGIONS_FRANCE = {
        "Champagne",
        "Côtes de Gascogne IGP",
        "Languedoc-Roussillon",
        "Sud-Ouest",
        "Vin de France AOC",
        "Corse",
        "Côtes du Lot Rocamadour IGP",
        "Bourgogne",
        "Rhône",
        "Beaujolais",
        "Provence-Alpes-Côte d'Azur",
        "Vin de France IGP",
        "Vin de France",
        "Bordeaux",
        "Loire",
        "Jura",
        "Alsace",
        "Savoie-Bugey",
    }
    df = df.with_columns(
        pl.col("characteristics")
        .str.split_exact("/", 1)
        .struct[1]
        .str.strip_chars()
        .alias("country")
    ).with_columns(
        pl.when(pl.col("country").is_in(REGIONS_FRANCE))
        .then(pl.lit("France"))
        .otherwise(pl.col("country"))
        .alias("country")
    )
    return df


def get_iso_country_code(df: pl.DataFrame) -> pl.DataFrame:
    """Permet de récupérer un code ISO à partir du nom d'un pays."""
    df = df.with_columns(
        pl.when(pl.col("country") == "France")
        .then(pl.lit("FRA"))
        .when(pl.col("country") == "Italie")
        .then(pl.lit("ITA"))
        .when(pl.col("country") == "Espagne")
        .then(pl.lit("ESP"))
        .when(pl.col("country") == "Argentine")
        .then(pl.lit("ARG"))
        .when(pl.col("country") == "Etats-Unis")
        .then(pl.lit("USA"))
        .when(pl.col("country") == "Australie")
        .then(pl.lit("AUS"))
        .when(pl.col("country") == "Chili")
        .then(pl.lit("CHL"))
        .when(pl.col("country") == "Allemagne")
        .then(pl.lit("DEU"))
        .when(pl.col("country") == "Afrique du Sud")
        .then(pl.lit("ZAF"))
        .when(pl.col("country") == "Portugal")
        .then(pl.lit("PRT"))
        .when(pl.col("country") == "Nouvelle-Zélande")
        .then(pl.lit("NZL"))
        .when(pl.col("country") == "Suisse")
        .then(pl.lit("CHE"))
        .when(pl.col("country") == "Autriche")
        .then(pl.lit("AUT"))
        .when(pl.col("country") == "Hongrie")
        .then(pl.lit("HUN"))
        .when(pl.col("country") == "Liban")
        .then(pl.lit("LBN"))
        .when(pl.col("country") == "Géorgie")
        .then(pl.lit("GEO"))
        .when(pl.col("country") == "Israël")
        .then(pl.lit("ISR"))
        .when(pl.col("country") == "Pérou")
        .then(pl.lit("PER"))
        .when(pl.col("country") == "Croatie")
        .then(pl.lit("HRV"))
        .when(pl.col("country") == "Grèce")
        .then(pl.lit("GRC"))
        .when(pl.col("country") == "Bulgarie")
        .then(pl.lit("BGR"))
        .when(pl.col("country") == "Slovénie")
        .then(pl.lit("SVN"))
        .when(pl.col("country") == "Syrie")
        .then(pl.lit("SYR"))
        .when(pl.col("country") == "Chine")
        .then(pl.lit("CHN"))
        .when(pl.col("country") == "Arménie")
        .then(pl.lit("ARM"))
        .when(pl.col("country") == "Roumanie")
        .then(pl.lit("ROU"))
        .when(pl.col("country") == "Uruguay")
        .then(pl.lit("URY"))
        .when(pl.col("country") == "Turquie")
        .then(pl.lit("TUR"))
        .when(pl.col("country") == "Angleterre")
        .then(pl.lit("GBR"))
        .when(pl.col("country") == "Maroc")
        .then(pl.lit("MAR"))
        .when(pl.col("country") == "Mexique")
        .then(pl.lit("MEX"))
        .otherwise(None)
        .alias("iso_code")
    )
    return df


def get_bubbles(df: pl.DataFrame) -> pl.DataFrame:
    """Permet de déterminer si un vin est effervescent ou non."""
    df = df.with_columns(
        pl.when(pl.col("type").str.contains("Effervescent"))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("bubbles")
    )
    return df


def get_new_type(df: pl.DataFrame) -> pl.DataFrame:
    """Doit etre éxécuté après get_bubbles."""
    df = df.with_columns(
        pl.col("type").str.replace("Effervescent", "Vin").alias("type")
    )
    return df


def get_cepage(df: pl.DataFrame) -> pl.DataFrame:
    """Extrait le cépage majoritaire."""
    df = (
        df.with_columns(
            pl.col("cepage")
            .str.split(",")
            .list.to_struct()
            .struct.rename_fields(["cepage"])
        )
        .unnest("cepage")
        .with_columns(
            pl.col("cepage")
            .str.extract_all(r"\D+")  # extrait tout sauf les digits
            .list.to_struct()
            .struct.rename_fields(["cepage"])
        )
        .unnest("cepage")
        .with_columns(pl.col("cepage").str.replace("% ", ""))
    )
    return df


def get_wine_note(df: pl.DataFrame) -> pl.DataFrame:
    """Extrait la note /20 associée au vin et la transforme en une note /5 ⭐."""
    df = df.with_columns(
        pl.col("note").str.slice(0, 4).cast(pl.Float64).truediv(4).alias("wine_note")
    )
    return df


def get_reviews(df: pl.DataFrame) -> pl.DataFrame:
    """Obtient le nombre de critiques des clients."""
    df = df.with_columns(
        pl.col("note")
        .str.replace(r"[0-9]+[.][0-9]+/20  noté par", "")
        .str.strip_chars()
        .str.replace(" clients", "")
        .cast(pl.Int64)
        .fill_null(strategy="zero")
        .alias("nb_reviews")
    )
    df = df.drop("note")
    return df


def get_service(df: pl.DataFrame) -> pl.DataFrame:
    """Réduit le nombre de modalités de service."""
    df = df.with_columns(
        pl.when(pl.col("service").str.contains("Passer en carafe"))
        .then(pl.lit("Passer en carafe"))
        .when(pl.col("service").str.contains("Ouvrir"))
        .then(pl.lit("Ouvrir avant le service"))
        .otherwise(pl.col("service"))
        .alias("service")
        .fill_null("Non Renseigné")
    )
    return df


def get_conservation_time(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("conservation_date").sub(pl.col("millesime")).alias("conservation_time")
    )
    return df


def is_cru(df: pl.DataFrame) -> pl.DataFrame:
    """Détermine si un vin est un grand cru ou non."""
    df = df.with_columns(
        pl.when(pl.col("name").str.contains("CRU"))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("cru")
    )
    return df


def drop_price(df: pl.DataFrame) -> pl.DataFrame:
    """Retire les prix non renseignés."""
    df = df.drop_nulls("unit_price")
    return df


def super_pipe(df: pl.DataFrame) -> pl.DataFrame:
    """`super_pipe`: Notre ours agrège l'ensemble des fonctions de cleaning 🐻.

    ---------
    `Parameters`
    --------- ::

        df (pl.DataFrame): # DataFrame d'entrée

    `Returns`
    --------- ::

        pl.DataFrame

    `Example(s)`
    ---------

    >>> super_pipe(df)
    ... #_test_return_"""
    df = (
        df.pipe(get_avg_temp)
        .pipe(get_valid_millesime)
        .pipe(get_conservation_date)
        .pipe(get_unit_and_offer_price)
        .pipe(get_keywords)
        .pipe(get_capacity)
        .pipe(get_bio)
        .pipe(get_customer_fav)
        .pipe(get_new)
        .pipe(get_top_100)
        .pipe(get_destock)
        .pipe(get_sulphite_free)
        .pipe(get_alcohol_volume)
        .pipe(get_cepage)
        .pipe(get_type)
        .pipe(get_country)
        .pipe(get_iso_country_code)
        .pipe(get_bubbles)
        .pipe(get_new_type)
        .pipe(get_wine_note)
        .pipe(get_reviews)
        .pipe(get_service)
        .pipe(get_conservation_time)
        .pipe(is_cru)
        .pipe(drop_price)
    )
    return df
