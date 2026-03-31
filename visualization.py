import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import dendrogram, linkage


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    corr = pd.read_csv(RESULTS_DIR / "model_correlation_matrix.csv", index_col=0)
    cosine = pd.read_csv(RESULTS_DIR / "model_cosine_similarity_matrix.csv", index_col=0)
    euclidean = pd.read_csv(RESULTS_DIR / "model_euclidean_distance_matrix.csv", index_col=0)
    performance = pd.read_csv(RESULTS_DIR / "model_performance_matrix.csv", index_col=0)
    return corr, cosine, euclidean, performance


def plot_heatmap(matrix, title, filename):
    plt.figure()
    sns.heatmap(matrix, annot=False)
    plt.title(title)
    plt.savefig(RESULTS_DIR / filename)
    plt.close()


def run_pca(data):
    pca = PCA(n_components=2)
    result = pca.fit_transform(data.values)

    plt.figure()
    plt.scatter(result[:, 0], result[:, 1])

    for i, name in enumerate(data.index):
        plt.text(result[i, 0], result[i, 1], name)

    plt.title("PCA of Model Performance")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.savefig(RESULTS_DIR / "pca.png")
    plt.close()


def run_tsne(data):
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    result = tsne.fit_transform(data.values)

    plt.figure()
    plt.scatter(result[:, 0], result[:, 1])

    for i, name in enumerate(data.index):
        plt.text(result[i, 0], result[i, 1], name)

    plt.title("t-SNE of Model Performance")

    plt.savefig(RESULTS_DIR / "tsne.png")
    plt.close()


def run_kmeans(data):
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(data.values)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data.values)

    plt.figure()
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels)

    for i, name in enumerate(data.index):
        plt.text(reduced[i, 0], reduced[i, 1], name)

    plt.title("KMeans Clustering of Models")

    plt.savefig(RESULTS_DIR / "kmeans.png")
    plt.close()


def run_dendrogram(data):
    linked = linkage(data.values, method='ward')

    plt.figure()
    dendrogram(linked, labels=data.index.tolist())

    plt.title("Hierarchical Clustering")

    plt.savefig(RESULTS_DIR / "dendrogram.png")
    plt.close()


def main():
    corr, cosine, euclidean, performance = load_data()

    plot_heatmap(corr, "Correlation Matrix", "correlation_heatmap.png")
    plot_heatmap(cosine, "Cosine Similarity Matrix", "cosine_heatmap.png")
    plot_heatmap(euclidean, "Euclidean Distance Matrix", "euclidean_heatmap.png")

    run_pca(performance)
    run_tsne(performance)
    run_kmeans(performance)
    run_dendrogram(performance)


if __name__ == "__main__":
    main()
