# src/clustering.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

DEFAULT_FEATURES = ["calories", "protein", "carbs", "fat", "iron", "vitamin_c"]

def run_kmeans(df, k=4, features=None):
    """
    Executa KMeans sobre as features fornecidas (ou DEFAULT_FEATURES) e adiciona coluna 'cluster'.
    Retorna (df_with_cluster, kmodel).
    """
    if features is None:
        features = DEFAULT_FEATURES

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Features ausentes para kmeans: {missing}")

    X = df[features].values
    kmodel = KMeans(n_clusters=k, random_state=42)
    labels = kmodel.fit_predict(X)
    df = df.copy()
    df['cluster'] = labels
    return df, kmodel

def compute_pca(df, n_components=2, features=None):
    """
    Executa PCA nas mesmas features (ou DEFAULT_FEATURES) e adiciona colunas pca1, pca2, ...
    Retorna (df_with_pca, pca_obj).
    """
    if features is None:
        features = DEFAULT_FEATURES

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Features ausentes para PCA: {missing}")

    pca = PCA(n_components=n_components, random_state=42)
    arr = pca.fit_transform(df[features].values)
    df = df.copy()
    for i in range(n_components):
        df[f"pca{i+1}"] = arr[:, i]
    return df, pca
