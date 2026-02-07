import pandas as pd

# kurze Daten übersicht
df = pd.read_csv("../datacsv/train.csv")

print("shape:", df.shape)
print("\nSpalten:", df.columns.tolist())
print("\nDatentypen:\n", df.dtypes)
print("\nFehlende Werte:\n", df.isna().sum())
print("\nDatum von/bis:", df["date"].min(), "->", df["date"].max())
print("\nErste Zeilen:\n", df.head())

# date in echtes Datum umwandeln
df["date"] = pd.to_datetime(df["date"])

# onpromotion bereinigen: NaN = False
df["onpromotion"] = df["onpromotion"].fillna(False).astype(bool)

print("\nNeue Datentypen:")
print(df["onpromotion"].value_counts())

# Zeit - Features aus dem Datum
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["weekday"] = df["date"].dt.weekday  # 0 = Montag, 6 = Sonntag

print("\nBeispiel mit Zeit-Features:")
print(df[["date", "year", "month", "weekday"]].head())

# Verkäufe nach Wochentag
weekday_sales = (
    df.groupby("weekday")["unit_sales"]
    .sum()
    .sort_index()
)
print("\nUnit sales pro Wochentag:")
print(weekday_sales)
# Der Wochentag hat einen starken Einfluss auf unit_sales
# weekday ist ein sehr starkes Feature
# Ein Modell MUSS diesen Effekt lernen
# Ohne Wochentag wäre jede Prognose schlechter


# oil-Daten laden
oil = pd.read_csv("../datacsv/oil.csv")

print(oil.head())
print(oil.describe())
