"""_summary_

    Returns:
        _type_: _description_
"""
import polars as pl
import re

def structure_clean(data):
    df = {
        "name" : data["name"].str.strip(),
        "capacity" : data["capacity"].str.strip(),
        "price" : data["prices"][0][0].strip(),
        "price_discount_1" : data["prices"][0][1].strip(),
        "price_discount_2" : data["prices"][0][2].strip(),
        "characteristic" : data["characteristic"].str.strip(),
        "note" : data["note"].str.strip(),
        "keyword_1": data["keyword"][0][0].strip(),
        "keyword_2": data["keyword"][0][1].strip(),
        "keyword_3": data["keyword"][0][2].strip(),
        "wine_classification": data["informations"][0][" \xa0\xa0Classification vin"].strip(),
        "vintage": data["informations"][0][" \xa0\xa0Millésime"].strip(),
        "grape_variety": data["informations"][0][' \xa0\xa0Cépage'].strip(),
        "taste": data["informations"][0][' \xa0\xa0Goûts'].strip(),
        "taste_type": data["informations"][0][' \xa0\xa0Par Goûts'].strip(),
        "eye": data["informations"][0][ " \xa0\xa0A l'oeil"].strip(),
        "nose": data["informations"][0][' \xa0\xa0Au nez'].strip(),
        "mouth": data["informations"][0][' \xa0\xa0En bouche'].strip(),
        "temperature": data["informations"][0][' \xa0\xa0Température de service'].strip(),
        "service": data["informations"][0][' \xa0\xa0Service'].strip(),
        "conservation": data["informations"][0][' \xa0\xa0Conservation'].strip(),
        "max_date": data["informations"][0][" \xa0\xa0Jusqu'à"].strip(),
        "food": data["informations"][0][' \xa0\xa0Accords mets-vin'].strip(),
        "recomandation": data["informations"][0][' \xa0\xa0Accords recommandés'].strip(),
        "picture" : data["picture"].str.strip()
    }
    return df

def clean_capacity(df):
    df = df.with_columns(
        pl.col("capacity").str.replace(" L","")
        .str.replace(",",".")
        .cast(pl.Float64)
    )
    return df

def clean_price(df):
    df = df.with_columns(
        pl.col("price").str.replace(" €","")
        .str.replace(",",".")
        .cast(pl.Float64),
    )
    return df

def clean_price_discount_1(df):
    try:
        df = df.with_columns(
            pl.col("price_discount_1").str.replace(" €","")
            .str.replace(",",".")
            .cast(pl.Float64),
        )
    except:
       df = df.with_columns(
            pl.when(pl.col("price_discount_1").str.lengths() == 4)
            .then(None).cast(pl.Float64).alias("price_discount_1")
       )
    
    return df

def clean_price_discount_2(df):   
    try:
        df = df.with_columns(
            pl.col("price_discount_2")
            .str.replace(" €","")
            .str.replace(",",".")
            .cast(pl.Float64),
        )
    except:
        df = df.with_columns(
            pl.when(pl.col("price_discount_2").str.lengths() == 4)
            .then(None).cast(pl.Float64).alias("price_discount_2")
       )
    
    return df

def clean_vintage(df):
    df = df.with_columns(
        pl.col("vintage").cast(pl.Int32)
    )
    return df

def clean_max_date(df):
    df = df.with_columns(
        pl.col("max_date").cast(pl.Int32)
    )
    return df

def clean_note(df):
    df = df.with_columns(
        pl.col("note").apply(
            lambda x: x.split()[0] if len(x.split()) > 0 else None
        ).alias("avis")
        .str.replace("/20", "")
        .cast(pl.Float32),
            
        pl.col("note").apply(
            lambda x: x.split()[-2] if len(x.split()) > 1 else None
        ).alias("nb_avis")
        .str.replace("/20", "")
        .cast(pl.Int32)
    )
        
    df = df.drop("note")
    return df


def extract_charac(text: str) -> int | None:
    """
    Extract the `year` of the wine from unstructured text using a regular expression pattern.
    """
    match = re.findall(r'[^/]+', text)
    if match:
        return match
    else:
        return None
    
def clean_characteristic(df):
    df = df.with_columns(
        pl.col("characteristic").apply(lambda x: extract_charac(x)[0]).alias("color"),
        pl.col("characteristic").apply(lambda x: extract_charac(x)[1]).alias("localisation"),
        pl.col("characteristic").apply(lambda x: extract_charac(x)[2]).alias("cru"),
        pl.col("characteristic").apply(lambda x: extract_charac(x)[3]).alias("degree")
    )
    df = df.drop("characteristic")
    return df

def extract_number(text: str) -> int | None:
    """
    Extract the `year` of the wine from unstructured text using a regular expression pattern.
    """
    match = re.findall(r'\d+', text)
    if match:
        return match
    else:
        return None
    
def clean_degree(df):
    df = df.with_columns(
        pl.col("degree").apply(lambda x: extract_number(x)[0]).cast(pl.Float32)
    )
    return df

def clean_service(df):
    try:
        df = df.with_columns(
            pl.col("service").apply(lambda x: extract_number(x)[0])
        )
    except:
        df = df.with_columns(
            pl.col("service").apply(lambda x: "None")
        )
    return df

def clean_temperature(df):
    df = df.with_columns(
        pl.col("temperature").apply(lambda x:extract_number(x)[0]).cast(pl.Float32)
    )
    return df

def clean_col(df):
    df = df.drop("taste")
    df = df.drop("eye")
    df = df.drop("nose")
    df = df.drop("mouth")
    df = df.drop("food")
    df = df.drop("recomandation")
    return df

def extract_cepages(text: str) -> str:
    """
    Extract the `name` of the wine from unstructured text.
    """
    match = re.split(r',\s*', text)
    if match:
        try:
            nouvelle_liste = [re.sub(r'\d+%', '', element).strip() for element in match]
            return nouvelle_liste
        except:
            return match
    else:
        return None
    
def clean_grape_variety(df):
    df = df.with_columns(
        pl.col("grape_variety").apply(lambda x:extract_cepages(x)[0])
    )
    return df

def clean_pipe(data):
    data_1 = structure_clean(data)
    df = pl.DataFrame(data_1)
    df = (
        df
        .pipe(clean_capacity)
        .pipe(clean_price)
        .pipe(clean_price_discount_1)
        .pipe(clean_price_discount_2)
        .pipe(clean_vintage)
        .pipe(clean_max_date)
        .pipe(clean_note)
        .pipe(clean_characteristic)
        .pipe(clean_service)
        .pipe(clean_temperature)
        .pipe(clean_col)
        .pipe(clean_grape_variety)
        .pipe(clean_degree)
    )
    return df