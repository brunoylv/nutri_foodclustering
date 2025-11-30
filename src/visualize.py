import matplotlib.pyplot as plt
import seaborn as sns

def plot_clusters(df):
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        data=df,
        x="pca1",
        y="pca2",
        hue="cluster",
        palette="Set2"
    )
    plt.title("Clusters de Alimentos (PCA)")
    plt.show()