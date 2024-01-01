from src.modules.ml_models.prediction import init, random_forest, boosting
import polars as pl

df = pl.DataFrame()

for EXPLIQUEE in ("type", "unit_price"):
    X_train_n, _, y_train, _, _ = init(EXPLIQUEE)
    X_train = X_train_n.drop(columns=["name"])

    for nom_model in ("Random Forest", "Boosting"):
        if nom_model == "Random Forest":
            model = random_forest(EXPLIQUEE, "Random Forest")
        elif nom_model == "Boosting":
            model = boosting(EXPLIQUEE, "Boosting")

        model.fit(X_train, y_train)

        importances = model.steps[-1][1].feature_importances_
        column_names = X_train.columns

        sorted_indices = sorted(range(len(importances)), key=lambda k: importances[k])
        sorted_importances = [importances[i] for i in sorted_indices]
        sorted_column_names = [column_names[i] for i in sorted_indices]

        temporaire = pl.DataFrame(
            {
                "importances": sorted_importances,
                "column_names": sorted_column_names,
                "id": len(importances) * [f"{EXPLIQUEE} {nom_model}"]
            }
        )

        df = pl.concat([df, temporaire])

df.write_csv("./data/importance.csv", separator=",")