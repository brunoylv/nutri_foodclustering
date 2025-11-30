import numpy as np

def compute_raw_score(df):
    df["nutri_raw"] = (
        df["protein"] * 2.0 +
        df["iron"] * 1.5 +
        df["vitamin_c"] * 1.5 -
        df["calories"] * 0.05 -
        df["fat"] * 1.0 -
        df["carbs"] * 0.5
    )
    return df

def normalize_score(df):
    min_val = df["nutri_raw"].min()
    max_val = df["nutri_raw"].max()
    df["nutri_score"] = ((df["nutri_raw"] - min_val) / (max_val - min_val)) * 100
    return df