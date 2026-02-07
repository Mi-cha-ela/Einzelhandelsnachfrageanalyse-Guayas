import pandas as pd

# Wir laden NUR die Spalten, die wir wirklich brauchen (schneller, weniger RAM)
usecols = ["date", "store_nbr", "unit_sales", "onpromotion"]

# Wir parsen date direkt als Datum (statt"Text")
# low_memory=False verhindert diese Mixed-Type-Warnungen öfter
df = pd.read_csv(
    "../datacsv/train.csv",
    usecols=usecols,
    parse_dates=["date"],
    low_memory=False
)

print("Shape:", df.shape)
print(df.head())

# onpromotion bereinigen (NaN = kein Angebot)
df["onpromotion"] = df["onpromotion"].fillna(False).astype(bool)

# unit_sales: negative Werte sind Retouren/Stornos -> fürs Forecasting meist auf 0 setzen
df["unit_sales"] = df["unit_sales"].clip(lower=0)

# --Train/Validation Split nach Datum --
cutoff = pd.Timestamp("2017-07-01")

df["weekday"] = df["date"].dt.weekday  # 0=Mo, 6=So

train = df[df["date"] < cutoff]
valid = df[df["date"] >= cutoff]

print("\nTRAIN:", train["date"].min(), "->", train["date"].max(), "rows:", len(train))
print("VALID:", valid["date"].min(), "->", valid["date"].max(), "rows:", len(valid))

# --- Baseline: gleiche Verkäufe wie vor 7 Tagen ---
df_sorted = df.sort_values(["store_nbr", "date"])

df_sorted["lag_7"] = (
    df_sorted
    .groupby("store_nbr")["unit_sales"]
    .shift(7)
)

valid_sorted = df_sorted[df_sorted["date"] >= cutoff].copy()
valid_sorted["lag_7"] = valid_sorted["lag_7"].fillna(0)

mae = (valid_sorted["unit_sales"] - valid_sorted["lag_7"]).abs().mean()
print("\nBaseline MAE (lag_7):", mae)

print("\nBeispiel (actual vs prediction):")
print(valid_sorted[["date", "store_nbr", "unit_sales", "lag_7"]].head())



# Durchschnittliche Verkäufe pro Store & Wochentag
weekday_mean = (
    train
    .groupby(["store_nbr", "weekday"])["unit_sales"]
    .mean()
    .rename("weekday_mean")
    .reset_index()
)

# Mit Validierungsdaten verbinden
valid_wd = valid.merge(
    weekday_mean,
    on=["store_nbr", "weekday"],
    how="left"
)

# falls etwas fehlt(selten):auf 0 setzen
valid_wd["weekday_mean"] = valid_wd["weekday_mean"].fillna(0)

# Neue MAE berechnen
mae_wd = (valid_wd["unit_sales"] - valid_wd["weekday_mean"]).abs().mean()
print("\nBaseline MAE (Wochentag_Durchschnitt):",mae_wd)

print("\nBeispiel:")
print(valid_wd[["date", "store_nbr", "weekday", "unit_sales", "weekday_mean"]].head())


