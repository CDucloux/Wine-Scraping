"""Et concrétement, ça donne quoi ton modèle ?"""
import polars as pl
from prediction import *

# TODO: utiliser pathlib c'est mieux quand même...
# TODO: éviter la syntaxe de pandas [] mais plutot utiliser la syntaxe de polars !

EXPLIQUEE = "unit_price"  # type or unit_price

if EXPLIQUEE == "type":
    MODE = "classification"
elif EXPLIQUEE == "unit_price":
    MODE = "regression"

X_train, X_test, y_train, y_test, data = init(EXPLIQUEE)
masque = data.index.isin(y_test.index)
df = data[masque]

model_rf = random_forest(EXPLIQUEE, "Random Forest")
model_boost = boosting(EXPLIQUEE, "Boosting")
model_ridge = ridge(EXPLIQUEE, "Ridge")
model_knn = knn(EXPLIQUEE, "K Neighbors")
model_mlp = mlp(EXPLIQUEE, "Réseaux de neurones")
model_sv = support_vector(EXPLIQUEE, "Support Vector")

models = [model_rf, model_boost, model_ridge, model_knn, model_mlp, model_sv]

for model in models:
    model.fit(X_train, y_train)

preds_rf = model_rf.predict(X_test).astype(str)
preds_boost = model_boost.predict(X_test).astype(str)
preds_ridge = model_ridge.predict(X_test).astype(str)
preds_knn = model_knn.predict(X_test).astype(str)
preds_mlp = model_mlp.predict(X_test).astype(str)
preds_sv = model_sv.predict(X_test).astype(str)

df = pl.DataFrame(data[masque])
df = df.select(
    pl.col("name"),
    pl.col(EXPLIQUEE),
    random_forest=pl.lit(preds_rf),
    boosting=pl.lit(preds_boost),
    ridge=pl.lit(preds_ridge),
    knn=pl.lit(preds_knn),
    mlp=pl.lit(preds_mlp),
    support_vector=pl.lit(preds_sv),
)

df.write_csv(f"./data/tables/pred_{MODE}.csv", separator=",")
