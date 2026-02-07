from pathlib import Path
import os, json # <--- Dateien speichern(MLflow)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error,mean_squared_error  # --> Modellbewertung
from sklearn.model_selection import RandomizedSearchCV       # Hyperparameter-Optimierung
from xgboost import XGBRegressor       # Mein Modell

import mlflow
import mlflow.sklearn # Dieses Modell mit diesen Parametern hatte diese Qualität

# ======= Projektpfade (ZENTRAL DEFINIERT – GANZ WICHTIG) ======


BASE_DIR = Path(__file__).resolve().parents[1]  #../favorita-grocery-sales-forecasting

DATA_DIR    = BASE_DIR / "data"
MODELS_DIR  = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MLRUNS_DIR  = BASE_DIR / "mlruns"

# Ordner automatisch erstellen
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
MLRUNS_DIR.mkdir(exist_ok=True)



# Einstellungen
DATA_PATH = DATA_DIR / "guayas_top3_q1_2014.pkl"  # <-- die Datei, die prepare_q1_2014.py erzeugt hat
EXPERIMENT_NAME = "favorita-guayas-xgb"

print("DATA_PATH:", DATA_PATH)
print("Exists?", DATA_PATH.exists())

# Metriken
def regression_metrics(y_true,y_pred, eps=1e-9):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true

    mae = np.mean(np.abs(err))       # Durchschnittlicher Fehler
    rmse = np.sqrt(np.mean(err**2))  # Fehler mit Fokus auf Ausreißer
    bias = np.mean(err)         # Systematische Über/Unterprognose
    mad = np.median(np.abs(err))  # robuster Fehler (Median)
    rmad = mad / (np.median(np.abs(y_true)) + eps)    # Relativer Fehler
    mape = np.mean(np.abs(err) / (np.abs(y_true) + eps)) * 100    # Prozentualer Fehler

    # MAPE ist hier hoch -> intermittierende Verkäufe (normal)

    return{"MAE": mae, "RMSE": rmse, "BIAS": bias, "MAD": mad, "rMAD": rmad, "MAPE": mape}

# Daten laden + Filter Q1 2014
# ------------------------------
df = pd.read_pickle(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

# falls noch mehr Familien drin wären, Sicherheitshalber :
top3 = ["GROCERY I", "BEVERAGES", "CLEANING"]
df= df[df["family"].isin(top3)].copy()
# Je höher die Aggregation, desto weniger Zufall, desto mehr Struktur
# Q1 2014
df = df[(df["date"] >= "2014-01-01") & (df["date"] <= "2014-03-31")].copy()

print("Shape Q1:", df.shape, "Zeitraum:", df["date"].min(), "->", df["date"].max())

# Features bauen (wie Woche 2 - minimal & robust)
# -----------------------------------------------
# erst sortieren, dann groupby, dann logs/rolling
df = df.sort_values(["store_nbr", "item_nbr", "date"]).copy()

df["weekday"] = df["date"].dt.weekday
df["month"] = df["date"].dt.month
df["is_weekend"] = (df["weekday"] >= 5).astype(int)
df["onpromotion"] = df["onpromotion"].fillna(False).astype(bool)
if "perishable" not in df.columns:
    df["perishable"] = 0
# nachfrage gleich an allen Tagen

# Lags pro (store,item)
grp = df.groupby(["store_nbr", "item_nbr"]) # jede Store x item getrennt
df["lag_1"] = grp["unit_sales"].shift(1)  ## Verkauf gestern
df["lag_7"] = grp["unit_sales"].shift(7)  ## Verkauf vor 7 Tagen    <-- Gedächtnis für das Modell
df["roll_mean_7"] = grp["unit_sales"].shift(1).transform(lambda s: s.rolling(7).mean()) # glättet zufallsschwankungen

print(df[["store_nbr","item_nbr","date","unit_sales","lag_1","roll_mean_7"]].head(10))



# NaNs nach shifts
df[["lag_1", "lag_7", "roll_mean_7"]] = df[["lag_1", "lag_7", "roll_mean_7"]].fillna(0.0)
# Anfang einer Serie hat keine Historie
# XGBoost mag keine NaN's

# One_hot für family (und optional city/type) - family reicht meist
df = pd.get_dummies(df, columns=["family"], drop_first=False)

# Chronologischer Split: Train Jan/Feb, Test März
# -----------------------------------------------
train = df[df["date"] < "2014-03-01"].copy()
test = df[df["date"] >= "2014-03-01"].copy()
# Zeitlich korrekt splitten, nie mischen - Januar/Februar -> lernen
# März -> testen

y_train = train["unit_sales"].values
y_test = test["unit_sales"].values
# Zielvariable = Unit_sales


# Features wählen (date raus, target raus)
drop_cols = ["date", "unit_sales"]
X_train = train.drop(columns=drop_cols)
X_test = test.drop(columns=drop_cols)

print("Train rows:", len(train), "Test rows:", len(test))
print("X_train:", X_train.shape, "X_test:", X_test.shape)


# Features-Spalten speichern (für Woche 4)
feature_cols = X_train.columns.tolist()
feature_path = MODELS_DIR / "feature_cols.json"

with open(feature_path, "w") as f:
    json.dump(feature_cols, f)
print("✅ feature_cols gespeichert:", feature_path)



# MLflow Setup
# --------------
mlflow.set_tracking_uri(f"file:{MLRUNS_DIR.resolve()}")
mlflow.set_experiment(EXPERIMENT_NAME)

# Baseline XGB trainieren + loggen = dict(
# -------------------------
baseline_params = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

baseline_model = XGBRegressor(**baseline_params)
baseline_model.fit(X_train, y_train)
pred_base = baseline_model.predict(X_test)
metrics_base = regression_metrics(y_test, pred_base)
print("Baseline metrics:", metrics_base)

# Plot speichern
plot_path = "../reports/baseline_pred_plot.png"
plt.figure()
plt.plot(y_test[:200], label="Ist")
plt.plot(pred_base[:200], label="Prognose")
plt.title("Baseline XGB: Ist vs Prognose (Ausschnitt)")
plt.legend()
plt.tight_layout()
baseline_plot = REPORTS_DIR / "baseline_pred_plot.png"
plt.savefig(baseline_plot, dpi=150)
plt.close()

with mlflow.start_run(run_name="baseline_xgb"):
    mlflow.log_params(baseline_params)
    mlflow.log_metrics({k: float(v) for k, v in metrics_base.items()})
    mlflow.log_artifact(str(baseline_plot))
    mlflow.log_artifact(str(feature_path))
    mlflow.sklearn.log_model(baseline_model, name="model")

# Tuning + Best Run..Hyperparameter-Raum
param_dist = {
    "n_estimators": [200, 400, 800],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [1, 5, 10],
    "reg_lambda": [1.0, 5.0, 10.0],
}
# RandomizedSearchCV starten
search = RandomizedSearchCV(
    estimator=XGBRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=20,
    scoring="neg_mean_absolute_error",
    cv=3,
    verbose=1,
    random_state=42
)
search.fit(X_train, y_train)
best_params = search.best_params_
print("Best params:", best_params)

# Bestes Model trainieren
best_model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
best_model.fit(X_train, y_train)

pred_best = best_model.predict(X_test)
metrics_best = regression_metrics(y_test, pred_best)
print("Best metrics:", metrics_best)

# Plot fürs beste Modell speichern
best_plot_path = REPORTS_DIR / "best_pred_plot.png"
plt.figure()
plt.plot(y_test[:200], label="Ist")
plt.plot(pred_best[:200], label="Prognose")
plt.title("Beste XGB: Ist vs Prognose (Ausschnitt)")
plt.legend()
plt.tight_layout()
plt.savefig(best_plot_path, dpi=150)
plt.close()

# Zweiten MLflow Run loggen
with mlflow.start_run(run_name="best_xgb_after_tuning"):
    mlflow.log_params(best_params)
    mlflow.log_metrics({k: float(v) for k, v in metrics_best.items()})
    mlflow.log_artifact(best_plot_path)
    mlflow.sklearn.log_model(best_model, name="model")

print("\nFertig. MLflow Logs in:", os.path.abspath(MLRUNS_DIR))
print("✅ Reports in:", REPORTS_DIR.resolve())


## Das Modell ist nicht perfekt, weil das Problem nicht perfekt vorhersagbar ist ##



