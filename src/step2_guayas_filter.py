import pandas as pd

# Stores laden und Guayas-Stores finden
stores = pd.read_csv("../datacsv/stores.csv")

guayas_stores = stores.loc[stores["state"].eq("Guayas"), "store_nbr"].unique()

print("Anzahl Stores in Guayas:", len(guayas_stores))
print("Guayas store_nbr (erste 20):", sorted(guayas_stores)[:20])
print("\nGuayas-Stores Übersicht:")
print(stores[stores["state"].eq("Guayas")][["store_nbr", "city", "type", "cluster"]].head(20))


# -- Train in Chunks laden + Guayas filtern --
usecols = ["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]

chunks = []
chunk_size = 1_000_000 # stabiler Startwert

for chunk in pd.read_csv(
        "../datacsv/train.csv",
    usecols=usecols,
    parse_dates=["date"],
    chunksize=chunk_size,
    low_memory=False,
):
    # nur Guayas
    chunk = chunk[chunk["store_nbr"].isin(guayas_stores)]

    # basic cleaning
    chunk["onpromotion"] = chunk["onpromotion"].fillna(False).astype(bool)
    chunk["unit_sales"] = chunk["unit_sales"].clip(lower=0)

    chunks.append(chunk)

df_guayas = pd.concat(chunks, ignore_index=True)

print("\nGuayas train shape:", df_guayas.shape)
print("Zeitraum:" , df_guayas["date"].min(), "->", df_guayas["date"].max())
print(df_guayas.head())

# Schritt 1: 300.000 Zeilen sampeln
# -- Sample fur schnelle Experimente (300k) __
df_sample = df_guayas.sample(n=300_000, random_state=42)

print("\nSample shape:", df_sample.shape)
print(df_sample.head())

# Schritt 2: items.csv laden (Artikel-Infos)
# Welche Produktfamilien sind in Guayas am wichtigsten?
# --items.csv laden --
items= pd.read_csv("../datacsv/items.csv")

print("\nItems shape:", items.shape)
print(items.head())

# Schritt 3: Sample mit Artikel-Infos verbinden
# Schlüssel ist item_nbr
# -- Sample mir items verbinden --
df_sample = df_sample.merge(
    items[["item_nbr", "family"]],
    on="item_nbr",
    how="left"
)

print("\nNach items_Merge:")
print(df_sample.head())

# Schritt 4: Top-3 Produktfamilien bestimmen "gemessen an der Anzahl der verschiedenen Artikel"
# wie viele verschiedene item_nbr pro family?
# --Top-3 Familien nach Anzahl unterschiedlicher Artikel --
top_families = (
    items
    .groupby("family")["item_nbr"]
    .nunique()
    .sort_values(ascending=False)
    .head(3)
)

print("\nTop-3 Produktfamilien:")
print(top_families)

# Zusammen decken sie sehr unterschiedliche Nachfrage - Dynamiken ab

# Sample auf nur diese Top-3 Familien begrenzen
top3_families = ["GROCERY I", "BEVERAGES", "CLEANING"]

df_sample_top3 = df_sample[df_sample["family"].isin(top3_families)]

print("\nSample (Guayas, Top-3 Familien) shape:", df_sample_top3.shape)
print(df_sample_top3["family"].value_counts())

# ---- Speichern für Woche 2 ----
df_sample_top3.to_pickle("guayas_top3_sample.pkl")
print("\nGespeichert als guayas_top3_sample.pkl")

# speichert die gefilterten Daten ohne die Chunk- Schleife erneut ausführen zu müssen