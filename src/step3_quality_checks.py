import pandas as pd

# vorbereitete Guayas-Daten laden
df = pd.read_pickle("../data/guayas_top3_sample.pkl")
print("Shape:", df.shape)
print(df.head())

# --- Fehlende Werte prüfen ---
missing = df.isna().sum()

print("\nFehlende Werte pro Spalte:")
print(missing)

# keine fehlende Werte/Datensatz sauber...

# Punkt 2: Fehlende Kalendertage prüfen & auffüllen
# jetzt kommt der wichtigste Zeitreihen-Schritt überhaupt:
# Jeder(Store,Item)muss eine durchgehende tägliche Zeitachse haben

# ----Wie viele Tage soll es geben?----
min_date = df["date"].min()
max_date = df["date"].max()

all_days = pd.date_range(min_date, max_date, freq="D")

print("Zeitraum:", min_date, "->", max_date)
print("Erwartete Anzahl Kalendertage:", len(all_days))
# Zeitachse sauber definiert

# Tatsächliche Tage pro(Store, Item)
# Haben alle (Store, Item)-Kombinationen wirklich alle Tage?
# ---- Anzahl vorhandener Tage ----
days_per_series =(
    df.groupby(["store_nbr", "item_nbr"])["date"]
      .nunique()
)

print("\nStatistik vorhandener Tage pro (Store, Item):")
print(days_per_series.describe())

# Punkt 3: EDA-erste einfache Visualisierung
# Verstehen, wie sich Verkäufe über die Zeit verhalten(Guayas, Top-3 Familien)
# EDA: Gesamtverkäufe über die Zeit --
daily_sales = (
    df.groupby("date")["unit_sales"]
      .sum()
      .reset_index()
)

print(daily_sales.head())
print(daily_sales.tail())

# --- Beispiel: Serie mit wenigen Tagen ---
example = days_per_series.sort_values().head(1)
store_ex, item_ex = example.index[0]

print("Beispiel-Serie:")
print("Store:", store_ex, "Item:", item_ex)
print("Tage vorhanden:", example.iloc[0])

df_example = df[(df["store_nbr"] == store_ex) & (df["item_nbr"] == item_ex)]
print(df_example.sort_values("date").head(10))
print("...")
print(df_example.sort_values("date").tail(10))

# Wochentag-Effekt analysieren
# nochmal sauber erklärt, warum die MAE vorhin so stark gesunken ist.
# Verkäufe nach Wochentag - Wochenrythmus, andere Nachfrage an Wochenenden und ruhige Tage unter der Woche
# ---Wochentag berechnen ----
df["weekday"] = df["date"].dt.weekday # 0=Mo, 6=So

# Zur Kontrolle
print(df[["date", "weekday"]].head())

# Jetzt summieren wir alle Verkäufe pro Wochentag
weekday_sales = (
    df.groupby("weekday")["unit_sales"]
    .sum()
    .reset_index()
)

print("\nUnit Sales pro Wochentag:")
print(weekday_sales)

# Diagram
import matplotlib.pyplot as plt

# Wenn weekday_sales ein DataFrame mit Spalten ist:
# (weekday, unit_sales)
weekday_sales_df = (
    df.groupby("weekday")["unit_sales"]
      .sum()
      .reset_index()
      .sort_values("weekday")
)

weekday_names = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

plt.figure()
plt.bar(weekday_names, weekday_sales_df["unit_sales"].to_numpy())
plt.title("Guayas (Sample): Unit Sales nach Wochentag")
plt.xlabel("Wochentag")
plt.ylabel("Summe unit_sales")
plt.tight_layout()
plt.show()


# ✔️ Guayas sauber gefiltert
#
# ✔️ Top-3 Produktfamilien bestimmt
#
# ✔️ Datenqualität geprüft
#
# ✔️ Zeitliche Struktur verstanden
#
# ✔️ Wochentag-Effekt nicht nur berechnet, sondern visualisiert

# Punkt 3 Promotion-Effekt(Guayas,Sample)
# Wir arbeiten weiter mir df aus guayas_top3_sample.pkl
# Durchschnittliche Verkäufe mit vs ohne Promotion
# --- Promotion-Effekt: Durchschnittliche Verkäufe ---
promo_stats = (
    df.groupby("onpromotion")["unit_sales"]
      .mean()
      .reset_index()
)

print("\nDurchschnittliche unit_sales:")
print(promo_stats)

# onpromotion = False -> Normalverkauf
# onpromotion = True  -> Angebotsverkauf

# balkendiagramm (sehr anschaulich)
import matplotlib.pyplot as plt

labels = ["Keine Promotion", "Promotion"]
values = promo_stats["unit_sales"].to_numpy()

plt.figure()
plt.bar(labels, values)
plt.title("Guayas (Sample): Durchschnittliche Verkäufe\nmit vs. ohne Promotion")
plt.ylabel("Durchschnitt unit_sales")
plt.tight_layout()
plt.show()

# Wenn der Balken „Promotion“ deutlich höher ist:

# 👉 Promotion wirkt stark

# 👉 erklärt viele Peaks

# 👉 MUSS ins Modell
# Mit Promotion werden im Schnitt fast doppelt so viele Einheiten verkauft
# + 5,3 Einheiten pro Tag,pro Artikel & Store - Ein sehr starker Effekt

# Promotion - Wochentag
# Wirkt eine Promotion an bestimmten Tagen stärker?
# Menschen kaufen nicht jeden Tag gleich/ Eine Promotion am Samstag bringt mehr als am Dienstag.

#🔹 Warum ist das wichtig fürs Modell?

# Wenn dein Modell weiß:

#„Samstag + Promotion = sehr hohe Nachfrage“
#…dann kann es:
#besser planen
#weniger Out-of-Stock
#weniger Verschwendung (wichtig bei perishable)
promo_weekday = (
    df
    .groupby(["weekday", "onpromotion"])["unit_sales"]
    .mean()
    .reset_index()
)

print(promo_weekday)

#🎯 Die eine zentrale Erkenntnis (bitte merken)
#Promotion wirkt nicht gleich stark.Sie wirkt am stärksten am Wochenende, besonders sonntags.
#Das ist:
#intuitiv ✔
#datenbasiert ✔
#extrem wertvoll fürs Modell ✔
# 🧠 Warum das später wichtig ist
# Ein gutes Modell kann lernen: Promotion + Sonntag → sehr hohe Nachfrage
# Promotion + Mittwoch → moderate Steigerung
# 👉 Genau solche Interaktionen machen den Unterschied.

import matplotlib.pyplot as plt

# Daten trennen
promo_yes = promo_weekday[promo_weekday["onpromotion"] == True]
promo_no  = promo_weekday[promo_weekday["onpromotion"] == False]

# Wochentagsnamen
weekday_names = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

plt.figure()
plt.plot(weekday_names, promo_no["unit_sales"].to_numpy(), label="Keine Promotion")
plt.plot(weekday_names, promo_yes["unit_sales"].to_numpy(), label="Promotion")

plt.title("Guayas (Sample): Promotion-Effekt nach Wochentag")
plt.xlabel("Wochentag")
plt.ylabel("Ø unit_sales")
plt.legend()
plt.tight_layout()
plt.show()

# Beide steigen richtung Wochenende

# In Guayas wirkt Promotion an allen Wochentagen positiv,
# jedoch ist der Effekt am Wochenende – insbesondere sonntags – am stärksten.


# Promotion * Produktfamilie
# Wirkt eine Promotion bei allen Produktfamilien gleich?
promo_family = (
    df
    .groupby(["family", "onpromotion"])["unit_sales"]
    .mean()
    .reset_index()
)
print(promo_family)

import matplotlib.pyplot as plt

# Daten aufteilen
promo_yes = promo_family[promo_family["onpromotion"] == True]
promo_no  = promo_family[promo_family["onpromotion"] == False]

families = promo_no["family"].to_list()

plt.figure()
plt.bar(families, promo_no["unit_sales"].to_numpy(), label="Keine Promotion")
plt.bar(families, promo_yes["unit_sales"].to_numpy(), bottom=promo_no["unit_sales"].to_numpy(), label="Promotion")

plt.title("Guayas (Sample): Promotion-Effekt nach Produktfamilie")
plt.ylabel("Ø unit_sales")
plt.legend()
plt.tight_layout()
plt.show()
# gestapelt damit man den Mehrwert durch Promotion direkt sieht

# 🥤 BEVERAGES – stärkste Reaktion
# impulsiv
# gut lagerbar
# stark preisgetrieben
# oft Vorratskäufe bei Angeboten
# 👉 klassischer Promotion-Gewinner

# 🧽 CLEANING – schwächste Reaktion
# Bedarfskäufe
# weniger spontan
# wird gekauft, wenn es nötig ist
# 👉 Promotion hilft, aber nicht explosiv

# onpromotion → sehr stark ✔
#
# weekday → klarer Zyklus ✔
#
# family → unterschiedliche Reaktionen ✔
#Für die Region Guayas zeigt die explorative Analyse einen starken Promotion-Effekt,
#der insbesondere am Wochenende und in der Produktfamilie BEVERAGES ausgeprägt ist.
# Wochentag und Produktfamilie erweisen sich als wichtige erklärende Variablen für die Nachfrage.

## Reflexion – Woche 1 (Guayas)

# In Woche 1 wurde der Datensatz auf die Region Guayas gefiltert und auf die drei größten Produktfamilien
# (GROCERY I, BEVERAGES, CLEANING) reduziert. Die explorative Analyse zeigt einen starken Einfluss von
# Promotionen auf die Nachfrage, insbesondere am Wochenende. Der Promotion-Effekt ist bei BEVERAGES
# am stärksten ausgeprägt, während CLEANING am wenigsten reagiert.

# Zusätzlich wurde ein klarer Wochentag-Effekt beobachtet, mit höheren Verkäufen am Samstag und Sonntag.
# Diese Erkenntnisse legen nahe, dass Promotion, Wochentag und Produktfamilie zentrale Features
# für die weitere Modellierung darstellen.

