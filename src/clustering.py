from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def run_kmeans(df, k=4):
    features = ["calories", "protein", "carbs", "fat", "iron", "vitamin_c"]

    kmeans = KMeans(n_clusters=k, random_state=42)
    df["cluster"] = kmeans.fit_predict(df[features])

    return df, kmeans

def apply_pca(df):
    pca = PCA(n_components=2)
    df[["pca1", "pca2"]] = pca.fit_transform(
        df[["calories","protein","carbs","fat","iron","vitamin_c"]]
    )
    return df