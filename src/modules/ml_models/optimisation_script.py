"""Script pour optimiser les paramètres des modèles."""
from models import *
from prediction import *
from sklearn.model_selection import train_test_split

EXPLIQUEE = "unit_price" #type or unit_price

if EXPLIQUEE == "type":
    CATEGORICALS = ["cepage", "par_gouts", "service", "country"]
    MODE = "classification"
elif EXPLIQUEE == "unit_price":
    CATEGORICALS = ["cepage", "par_gouts", "service", "country", "type"]
    MODE = "regression"
    
df = data_model(chemin= "./data/vins.json",
                variable_a_predire= EXPLIQUEE)

df = df.select("capacity", "unit_price","millesime", "cepage", "par_gouts",
          "service", "avg_temp", "conservation_date", "bio", "customer_fav", 
          "is_new", "top_100", "destock", "sulphite_free", "alcohol_volume",
          "country", "bubbles", "wine_note", "nb_reviews", "conservation_time", "cru",
          "type")

df = prep_str(df, categorical_cols=CATEGORICALS)

X = df.drop(columns=[EXPLIQUEE])
y = df[EXPLIQUEE]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = train_model(X_train, y_train, MODE)

stockage_result_csv(models, MODE)