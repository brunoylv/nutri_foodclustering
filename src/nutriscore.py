# src/nutriscore.py
import pandas as pd
import numpy as np

def compute_raw_score(df, weights=None):
    """
    Computa uma pontuação linear 'nutri_raw' a partir das colunas nutricionais.
    weights: dict opcional com pesos por nutriente. Valores positivos aumentam a pontuação.
    Retorna dataframe com coluna 'nutri_raw'.
    """
    df = df.copy()
    # pesos padrão (ajuste conforme desejar)
    if weights is None:
        weights = {
            'calories': -1.0,
            'fat': -0.8,
            'carbs': -0.6,
            'protein': 1.0,
            'iron': 0.5,
            'vitamin_c': 0.5
        }

    # garantir existência das colunas
    for k in weights.keys():
        if k not in df.columns:
            df[k] = 0.0

    # converter para float e calcular soma ponderada
    s = 0.0
    for col, w in weights.items():
        s = s + w * df[col].astype(float)

    df['nutri_raw'] = s
    return df

def normalize_score(df, feature='nutri_raw', out='nutri_score'):
    """
    Normaliza a coluna `feature` para a escala 0-100 e salva em `out`.
    Retorna dataframe com coluna `out`.
    """
    df = df.copy()
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' não encontrada no dataframe para normalização.")

    arr = df[feature].astype(float).values
    # lidar com NaNs
    mask = ~np.isnan(arr)
    if mask.sum() == 0:
        # nada para normalizar
        df[out] = 0.0
        return df

    amin = np.nanmin(arr)
    amax = np.nanmax(arr)
    if np.isclose(amax, amin):
        # todos iguais -> atribui valor médio (50)
        df[out] = 50.0
    else:
        norm = (arr - amin) / (amax - amin) * 100.0
        # conserva NaNs caso existam
        df[out] = np.where(mask, norm, np.nan)

    return df