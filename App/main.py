import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import DATA_PATH, MODELS_DIR, MLFLOW_TRACKING_URI, RUN_ID
from data.data_utils import load_base_df, make_features, align_to_feature_cols
from models.model_utils import load_feature_cols, load_mlflow_model
df = load_base_df(DATA_PATH)


# Streamlit Page Setup
# --------------------
st.set_page_config(page_title="Guayas Demand Forecaster", layout="wide")
st.title("Guayas Nachfrageprognose (Jan–Mar 2014)")
st.markdown("""
Diese App zeigt tägliche Nachfrageprognosen für Stores in **Guayas**.  
Wähle links eine Filiale und SKU/Familie und erhalte einen N-Tage Forecast.
""")

# Daten laden
df_raw = load_base_df(DATA_PATH)
# Spaltennamen säubern (ENTFERNT LEERZEICHEN!)
df_raw.columns = df_raw.columns.str.strip()
# 2) Alles auf ein internes Standard-Schema mappen
rename_map = {
    "Datum": "date",
    "date": "date",

    "Filialnr.": "store_nbr",
    "store_nbr": "store_nbr",

    "Artikelnr.": "item_nbr",
    "item_nbr": "item_nbr",

    "Stückzahlen": "unit_sales",
    "unit_sales": "unit_sales",

    "Aktion": "onpromotion",
    "onpromotion": "onpromotion",

    "Familie": "family",
    "Family": "family",
    "family": "family",
}

df_raw = df_raw.rename(columns=rename_map)
df_raw["date"] = pd.to_datetime(df_raw["date"])
# Nur Q1 2014 (wichtig für das Projektziel)
df_q1 = df_raw[(df_raw["date"] >= "2014-01-01") & (df_raw["date"] <= "2014-03-31")].copy()




# Sidebar (links)
# --------------
st.sidebar.header("⚙ Einstellungen")
st.sidebar.markdown("""
**Zielgruppe:** Bedarfsplaner in Guayas  
**Zeitraum:** Jan–Mär 2014  
Wähle Modus, Filiale und ggf. Familie/SKU, Startdatum und Prognosehorizont.
""")
mode = st.sidebar.selectbox(
    "Modus",
    ["Store gesamt", "Store + Familie", "Store + SKU (Bonus)"]
)

stores = sorted(df["store_nbr"].unique().tolist())
store = st.sidebar.selectbox("Filiale", stores)

show_actual = st.sidebar.checkbox("Ist-Werte anzeigen (falls vorhanden)", value=True)

min_date = pd.Timestamp("2014-01-01").date()
max_date = pd.Timestamp("2014-03-31").date()
start_date = st.sidebar.date_input("Startdatum", value=min_date, min_value=min_date, max_value=max_date)
n_days = st.sidebar.slider("N-Tage", min_value=1, max_value=30, value=7)


# Welche Spalte enthält die Produkt-Familie?
if "family" in df_q1.columns:
    FAMILY_COL = "family"
elif "Familie" in df_q1.columns:
    FAMILY_COL = "Familie"
else:
    FAMILY_COL = None

if FAMILY_COL is None:
    st.sidebar.warning("⚠️ Keine Spalte für Produkt-Familie im Datensatz gefunden.")
    mode = "Filiale + SKU"  # fallback
else:
    fams = sorted(df_q1.loc[df_q1["store_nbr"] == store, FAMILY_COL].dropna().unique().tolist())
    fam = st.sidebar.selectbox("Familie", fams, key="family_select")

# Build "series" depending on mode
# series muss am Ende Spalten haben: date, unit_sales(und optional store_nbr/item_nbr für Debug)

series = None

if mode == "Store gesamt":
    tmp = df_q1[df_q1["store_nbr"] == store].copy()
    series = tmp.groupby("date", as_index=False)["unit_sales"].sum()
    series["store_nbr"] = store
    series["item_nbr"] = -1

elif mode == "Store + Familie":
    fams = sorted(df_q1.loc[df_q1["store_nbr"] == store, "family"].dropna().unique().tolist())
    fam = st.sidebar.selectbox("Familie", fams)
    tmp = df_q1[(df_q1["store_nbr"] == store) & (df_q1["family"] == fam)].copy()
    series = tmp.groupby("date", as_index=False)["unit_sales"].sum()
    series["store_nbr"] = store
    series["item_nbr"] = -1

elif mode == "Store + SKU (Bonus)":
    # Bonus nur anzeigen, wenn genügend Punkte existieren
    counts = (
        df_q1[df_q1["store_nbr"] == store]
        .groupby("item_nbr")["date"]
        .nunique()
        .reset_index(name="n_days")
        .sort_values("n_days", ascending=False)
    )

    MIN_DAYS = st.sidebar.slider("Min. Datentage (Bonus-SKU)", 5, 60, 20)

    valid_skus = counts[counts["n_days"] >= MIN_DAYS]["item_nbr"].tolist()

    if not valid_skus:
        st.warning("⚠️ Für diese Filiale gibt es keine SKU mit genug Datentagen. Wechsel Filiale oder MIN_DAYS senken.")
        st.stop()

    sku = st.sidebar.selectbox("Artikel (SKU)", valid_skus)
    n_days_series = int(counts[counts["item_nbr"] == sku]["n_days"].iloc[0])
    st.sidebar.caption(f"📌 Diese SKU hat {n_days_series} Datentage in Q1 2014.")

    series = df_q1[(df_q1["store_nbr"] == store) & (df_q1["item_nbr"] == sku)].copy()
    series = series[["date", "unit_sales", "store_nbr", "item_nbr"]].copy()


# Sanity checks
# -------------
series = series.sort_values("date")
st.caption(f"Zeitraum in Auswahl: {series['date'].min().date()} → {series['date'].max().date()} | Punkte: {series['date'].nunique()}")

# Hinweis wenn Lücken
expected_days = pd.date_range(series["date"].min(), series["date"].max(), freq="D").size
have_days = series["date"].nunique()
missing = expected_days - have_days
if missing > 0:
    st.info(f"ℹ️ Hinweis: Es gibt {missing} fehlende Tage in dieser Auswahl (Lücken). Das ist normal bei dünnen SKU-Daten.")


# Model laden + feature cols
# --------------------------
feature_cols_path = MODELS_DIR / "feature_cols.json"
feature_cols = load_feature_cols(feature_cols_path)

try:
    model = load_mlflow_model(RUN_ID, MLFLOW_TRACKING_URI)
except Exception:
    st.error("❌ Modell konnte nicht geladen werden. Details:")
    st.code(traceback.format_exc())
    st.stop()


# Feature engineering + predict on existing dates
# (Sprint4-konform: historische Vorhersage innerhalb Jan–Mär 2014)
# ----------------------------------------------------------------
series_feat = make_features(series)

drop_cols = ["date", "unit_sales"]
X = series_feat.drop(columns=drop_cols, errors="ignore")
y_true = series_feat["unit_sales"].values if "unit_sales" in series_feat.columns else None
dates = pd.to_datetime(series_feat["date"].values)

X_aligned = align_to_feature_cols(X, feature_cols)

y_pred = model.predict(X_aligned)

out = pd.DataFrame({"date": dates, "y_pred": y_pred})
if y_true is not None and show_actual:
    out["y_true"] = y_true

# N-Tage Fenster ab Startdatum
start_ts = pd.Timestamp(start_date)
out = out.sort_values("date")
out_window = out[out["date"] >= start_ts].head(n_days).copy()

if out_window.empty:
    st.warning("⚠️ Ab diesem Startdatum gibt es in der Auswahl keine Datenpunkte. Wähle ein früheres Startdatum.")
    st.stop()



# Layout: Plot + Table
# --------------------
col1, col2 = st.columns([2, 1])

# Layout : Chart + KPIs + Tabelle
# -------------------------------

col_plot, col_side = st.columns([3, 1])


# PLOT (groß)
# =======================
with col_plot:

    st.subheader("📈 Forecast")

    fig, ax = plt.subplots()

    ax.bar(out_window["date"], out_window["y_pred"])
    ax.set_title("Forecasted Unit Sales")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units")

    plt.xticks(rotation=45)

    st.pyplot(fig)




# KPIs + Tabelle (rechts)
# =======================
with col_side:

    st.subheader("📊 Summary")

    total_pred = int(out_window["y_pred"].sum())
    mean_pred = round(out_window["y_pred"].mean(), 2)

    st.metric("Total Forecast", total_pred)
    st.metric("Ø Daily Sales", mean_pred)
    st.metric("Days", len(out_window))


    st.subheader("Details")
    st.dataframe(out_window, use_container_width=True)

    csv = out_window.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download CSV",
        csv,
        "forecast.csv",
        "text/csv"
    )


    # Bonus: Wochen-Bar-Chart
    # ------------------------
fig, ax = plt.subplots(figsize=(9,4))

ax.set_facecolor("#FFFFFF")

if "y_true" in out_window.columns and show_actual:
    ax.plot(
        out_window["date"],
        out_window["y_true"],
        label="Ist",
        linestyle="--",
        linewidth=2
    )

ax.plot(
    out_window["date"],
    out_window["y_pred"],
    label="Prognose",
    linewidth=3
)

ax.grid(alpha=0.25)
ax.legend()

st.pyplot(fig)
