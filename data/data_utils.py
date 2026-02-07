import pandas as pd
import numpy as np

def load_base_df(data_path) -> pd.DataFrame:
    df = pd.read_pickle(data_path)
    df["date"] = pd.to_datetime(df["date"])
    return df

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Baut die gleichen Features wie im Training:
    weekday, month, is_weekend, onpromotion, perishable, lag_1, lag_7, roll_mean_7
    + one-hot family
    """
    if "onpromotion" not in df.columns:
        df["onpromotion"] = False

    if "family" not in df.columns:
        df["family"] = "UNKNOWN"
    df = df.sort_values(["store_nbr", "item_nbr", "date"]).copy()

    df["weekday"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["onpromotion"] = df["onpromotion"].fillna(False).astype(bool)
    df["perishable"] = df.get("perishable", 0)

    grp = df.groupby(["store_nbr", "item_nbr"])
    df["lag_1"] = grp["unit_sales"].shift(1)
    df["lag_7"] = grp["unit_sales"].shift(7)
    df["roll_mean_7"] = grp["unit_sales"].shift(1).transform(lambda s: s.rolling(7).mean())

    df[["lag_1", "lag_7", "roll_mean_7"]] = df[["lag_1", "lag_7", "roll_mean_7"]].fillna(0.0)

    # One-hot family
    df = pd.get_dummies(df, columns=["family"], drop_first=False)

    return df

def align_to_feature_cols(X: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Sorgt dafür, dass X exakt die Spalten hat wie beim Training:
    - fehlende Spalten hinzufügen (0)
    - extra Spalten entfernen
    - Reihenfolge anpassen
    """
    X = X.copy()

    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_cols]
    return X
