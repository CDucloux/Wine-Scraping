import duckdb
from pathlib import Path

root = Path(".").resolve()
data_folder = root / "data"
wine_db_path = str(data_folder / "DB" / "dt.db")
tables_folder = str(data_folder / "tables")

conn = duckdb.connect(wine_db_path)

conn.sql(
    f"CREATE OR REPLACE TABLE pred_classification AS SELECT * FROM read_csv_auto('{tables_folder}\pred_classification.csv', header = true);"
)

conn.sql(
    f"CREATE OR REPLACE TABLE pred_regression AS SELECT * FROM read_csv_auto('{tables_folder}\pred_regression.csv', header = true);"
)

print(conn.execute("DESCRIBE ALL TABLES").pl())
