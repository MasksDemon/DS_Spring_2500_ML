import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import dendrogram, linkage


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    performance = pd.read_csv(RESULTS_DIR / "model_performance_matrix.csv", index_col=0)
    return performance


def run_pca(data):
    pca = PCA(n_components=2)
    result = pca.fit_transform(data.values)

    plt.figure()
    plt.scatter(result[:, 0], result[:, 1])

    for i, name in enumerate(data.index):
        plt.text(result[i, 0], result[i, 1], name)

    plt.title("PCA of Models")
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

    plt.title("t-SNE of Models")

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

    plt.title("KMeans Clustering")

    plt.savefig(RESULTS_DIR / "kmeans.png")
    plt.close()


def run_dendrogram(data):
    linked = linkage(data.values, method='ward')

    plt.figure()
    dendrogram(linked, labels=data.index.tolist())

    plt.title("Hierarchical Clustering Dendrogram")

    plt.savefig(RESULTS_DIR / "dendrogram.png")
    plt.close()


def main():
    data = load_data()

    run_pca(data)
    run_tsne(data)
    run_kmeans(data)
    run_dendrogram(data)


if __name__ == "__main__":
    main()
