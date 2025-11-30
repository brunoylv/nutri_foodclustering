from preprocess import load_dataset, preprocess
from nutriscore import compute_raw_score, normalize_score
from clustering import run_kmeans, apply_pca
from visualize import plot_clusters

def run_pipeline():
    df = load_dataset()

    df = preprocess(df)
    
    df = compute_raw_score(df)
    df = normalize_score(df)

    df, model = run_kmeans(df, k=4)
    df = apply_pca(df)

    print(df.head())

    plot_clusters(df)

if __name__ == "__main__":
    run_pipeline()