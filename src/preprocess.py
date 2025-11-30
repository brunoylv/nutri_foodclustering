# src/preprocess.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

NUTRIENT_COLS = ["calories", "protein", "carbs", "fat", "iron", "vitamin_c"]

def load_dataset(path="data/food_nutrition_dataset.csv"):
    """
    Carrega CSV do caminho informado e retorna um DataFrame.
    """
    df = pd.read_csv(path)
    return df

def preprocess(df):
    """
    Recebe dataframe bruto, faz coercion das colunas nutricionais para numérico,
    drop de linhas completamente nulas nas features nutricionais e aplica MinMaxScaler
    nas colunas definidas em NUTRIENT_COLS.
    Retorna um dataframe (cópia) com as colunas normalizadas entre 0 e 1.
    """
    df = df.copy()
    # garantir colunas numéricas
    for c in NUTRIENT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            # cria coluna com zeros se ausente (evita erro)
            df[c] = 0.0

    # remover linhas com todos os nutrientes NaN
    df = df.dropna(subset=NUTRIENT_COLS, how='all').reset_index(drop=True)

    scaler = MinMaxScaler()
    try:
        df[NUTRIENT_COLS] = scaler.fit_transform(df[NUTRIENT_COLS])
    except Exception:
        # fallback: se scaler falhar (ex: todas colunas constantes), apenas preencher NaNs por 0
        df[NUTRIENT_COLS] = df[NUTRIENT_COLS].fillna(0.0)

    return df