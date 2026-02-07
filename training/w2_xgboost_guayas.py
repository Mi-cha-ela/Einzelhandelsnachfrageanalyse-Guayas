import pandas as pd

# Daten laden (Woche 1 Ergebnis)
df = pd.read_pickle("../data/guayas_top3_sample.pkl")

# Check: Grösse + Spalten
print("Shape:", df.shape)
print("Spalten:", list(df.columns))

# Check: Datum-Bereich
df["date"] = pd.to_datetime(df["date"])
print("Zeitraum:", df["date"].min(), "->", df["date"].max())

print(df.head())


promo_effect = df.groupby("onpromotion")["unit_sales"].mean()
print(promo_effect)

# Zeitfenster festlegen
start_date = "2014-01-01"
end_date = "2014-04-01"  # exklusive -> bis 31.03.2014

df_q1 = df[(df["date"] >= start_date) & (df["date"] < end_date)]

#  Checks
print("Shape nach Zeitfilter:", df_q1.shape)
print("Zeitraum:", df_q1["date"].min(), "->", df_q1["date"].max())
print(df_q1.head())

# Train/Test Split
split_date = "2014-03-01"

df_train = df_q1[df_q1["date"] < split_date]
df_test  = df_q1[df_q1["date"] >= split_date]

# --- Metadaten laden ----
stores = pd.read_csv("../datacsv/stores.csv")
items = pd.read_csv("../datacsv/items.csv")

# ----Metadaten mergen ----
df_train = df_train.merge(
    stores[["store_nbr", "city", "type", "cluster"]],
    on="store_nbr",
    how="left"
)

df_train = df_train.merge(
    items[["item_nbr", "family", "class", "perishable"]],
    on="item_nbr"
    ,how="left"
)

df_test = df_test.merge(
    stores[["store_nbr", "city", "type", "cluster"]],
    on="store_nbr",
    how="left"
)
df_test = df_test.merge(
    items[["item_nbr", "family", "class", "perishable"]],
    on="item_nbr"
    ,how="left"
)

print("Train columns:", df_train.columns.tolist())
print("Test columns:", df_test.columns.tolist())
print(df_train.head())

# ------- Datumfeatures bauen -------
for d in [df_train, df_test]:
    d["weekday"] = d["date"].dt.weekday    # Mo=0 ... So=6
    d["is_weekend"] = d["weekday"].isin([5, 6]).astype(int)
    d["month"] = d["date"].dt.month

print(df_train[["date", "weekday", "is_weekend", "month"]].head(10))

# Vorbereitung: sortieren. Lags funktionieren nur bei richtiger Reihenfolge.
df_train = df_train.sort_values(["store_nbr", "item_nbr", "date"])
df_test = df_test.sort_values(["store_nbr", "item_nbr", "date"])

# Lag Features erzeugen
group_cols = ["store_nbr", "item_nbr"]

for d in [df_train, df_test]:
    g = d.groupby(group_cols)
    d["lag_1"] = g["unit_sales"].shift(1)
    d["lag_7"] = g["unit_sales"].shift(7)

# Fehlende Lags auffüllen
# Am Anfang jeder Serie gibt es keine Vergangenheit -> NaN

for d in [df_train, df_test]:
    d["lag_1"] = d["lag_1"].fillna(0)
    d["lag_7"] = d["lag_7"].fillna(0)

print(df_train[["date", "store_nbr", "unit_sales", "lag_1", "lag_7"]].head(10))

# -----Rolling Mean (7 Tage)----
# nimm die letzten Tage und bilde den Durchschnitt
for d in [df_train, df_test]:
    d["roll_mean_7"] = (
        d
        .groupby(["store_nbr", "item_nbr"])["unit_sales"]
        .shift(1)  # wichtig: NICHT den heutigen Tag nutzen
        .rolling(7)
        .mean()
    )
for d in [df_train, df_test]:
    d["roll_mean_7"] = d["roll_mean_7"].fillna(0)

    print(
        df_train[
            ["date", "store_nbr", "unit_sales", "lag_1", "lag_7", "roll_mean_7"]
        ].head(15)
    )
# Rolling Means sind aufgrund intermittierender Serien häufig 0.

# --------- Features & Target trennen ----------
from sklearn.model_selection import train_test_split

# Zielvariable
y_train = df_train["unit_sales"]
y_test  = df_test["unit_sales"]

# Feature-Spalten (bewusst ausgewählt)
feature_cols = [
    "store_nbr",
    "item_nbr",
    "onpromotion",
    "weekday",
    "is_weekend",
    "month",
    "lag_1",
    "lag_7",
    "roll_mean_7",
    "perishable",
    "class",
    "cluster"
]

X_train = df_train[feature_cols]
X_test  = df_test[feature_cols]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# XGBoost trainieren
# Was ist XGBoost ? Ein Entscheidungsbaum-Modell, sehr stark für tabellarische Daten!

# Modell erstellen & fitten
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)
model.fit(X_train, y_train)

print("Modelltraining abgeschlossen")

# Vorhersagen erzeugen
y_pred = model.predict(X_test)
print("Vorhersagen erstellt:", y_pred[:5])

# Beweis zum ansehen
check = df_test[["date", "store_nbr", "item_nbr",  "unit_sales"]].copy()
check["prediction"] = y_pred

print(check.head(10))

##„Ziel des Modells ist die Prognose der täglichen Verkaufsmenge (unit_sales) pro Store–Item-Kombination.“

# An welchen Tagen liegt das Modell daneben und in welche Richtung?
# Fehler pro Tag visualisieren
# wir vergleichen prediction - unit_sales - Wir aggregieren pro Datum
result = df_test[["date", "unit_sales"]].copy()
result["prediction"] = y_pred
result["error"] = result["prediction"] - result["unit_sales"]

print(result.head( ))

# Wir mitteln die Fehler über alle Artikel eines Tages:
daily_error = (
    result
    .groupby("date")["error"]
    .mean()
)

print(daily_error.head())

import matplotlib.pyplot as plt

plt.figure()
daily_error.plot()
plt.axhline(0) # Nulllinie
plt.title("Durchschnittlicher Vorhersagenfehler pro Tag (Guayas)")
plt.xlabel("Datum")
plt.ylabel("Prediction - Actual (unit_sales)")
plt.show()

# Welche Variablen haben die größte Wirkung? Warum trifft das Modell diese Entscheidung?
# Feature Importance
import pandas as pd

importance = pd.Series(
    model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print(importance)


import matplotlib.pyplot as plt

plt.figure()
importance.plot(kind="bar")
plt.title("Feature Importance - XGBoost (Guayas)")
plt.xlabel("Feature")
plt.ylabel("Wichtigkeit")
plt.show()

# Das Modell lernt viel über : "Welcher Artikel verkauft sich generell wie?"
#                              "Welcher Store ist groß/klein?"
# Aber weniger über: echte zeitliche Dynamik
#                    Wiederverkaufszyklen
#                    Promotion-Impulse

# ✅ Plan (LSTM schnell, sinnvoll, vergleichbar)
# Daten pro Tag & Family aggregieren
# Ein kleines sequenz-Datenset bauen(z.B.14 Tage -> nächster Tag)
# LSTM trainieren
# MAE/RMSE vergleichen mit XGBoost( auf derselben Testperiode)

# ACF- Analyse der Zeitreihe
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

# Tagesaggregation (Guayas Q1 2014)
ts= (
    df_q1
    .groupby("date")["unit_sales"]
    .sum()
    .sort_index()
)

# Differenzierte Zeitreihe (optional, aber empfohlen)
ts_diff = ts.diff().dropna()

plt.figure()
plot_acf(ts_diff, lags=30)
plt.title("ACF der differenzierten Verkaufszeitreihe (Guayas, Q1 2014)")
plt.tight_layout()
plt.show()

# Lag 7 ist sinnvoll -> wöchentliche Struktur sichtbar
# Lag 1 stark -> kurzfristige Abhängigkeit
# Rolling Mean sinnvoll -> Glättung bestätigt


