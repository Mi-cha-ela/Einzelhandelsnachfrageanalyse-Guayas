import pandas as pd

df = pd.read_pickle("../data/guayas_top3_sample.pkl")
df["date"] = pd.to_datetime(df["date"])

df_q1 = df[(df["date"] >= "2014-01-01") & (df["date"] <= "2014-03-31")].copy()
print("Q1 rows:", len(df_q1))

df_q1.to_pickle("guayas_top3_q1_2014.pkl")
print("Gespeichert als guayas_top3_q1_2014.pkl")
