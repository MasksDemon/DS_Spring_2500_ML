import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import dendrogram, linkage


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_results():
    """Load all result CSV files from the results directory."""
    corr = pd.read_csv(RESULTS_DIR / "model_correlation_matrix.csv", index_col=0)
    cosine = pd.read_csv(RESULTS_DIR / "model_cosine_similarity_matrix.csv", index_col=0)
    euclidean = pd.read_csv(RESULTS_DIR / "model_euclidean_distance_matrix.csv", index_col=0)
    performance = pd.read_csv(RESULTS_DIR / "model_performance_matrix.csv", index_col=0)

    h2h = None
    win = None

    if (RESULTS_DIR / "model_head_to_head.csv").exists():
        h2h = pd.read_csv(RESULTS_DIR / "model_head_to_head.csv", index_col=0)

    if (RESULTS_DIR / "model_win_counts.csv").exists():
        win = pd.read_csv(RESULTS_DIR / "model_win_counts.csv", index_col=0)

    return corr, cosine, euclidean, performance, h2h, win


def plot_heatmaps(corr, cosine, euclidean):
    """Generate heatmaps for similarity matrices."""
    def draw(matrix, title, filename):
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, cmap="coolwarm")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / filename)
        plt.close()

    draw(corr, "Correlation Heatmap", "model_correlation_heatmap.png")
    draw(cosine, "Cosine Similarity Heatmap", "model_cosine_heatmap.png")
    draw(euclidean, "Euclidean Distance Heatmap", "model_euclidean_heatmap.png")


def plot_kmeans(performance):
    """Apply PCA + KMeans to visualize model clusters."""
    X = performance.values.T
    model_names = performance.columns

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_pca)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)

    for i, name in enumerate(model_names):
        plt.text(X_pca[i, 0], X_pca[i, 1], name)

    plt.title("KMeans Clustering (PCA Projection)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "kmeans_clusters.png")
    plt.close()


def plot_dendrogram(performance):
    """Generate hierarchical clustering dendrogram."""
    X = performance.values.T
    model_names = performance.columns

    Z = linkage(X, method="ward")

    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=model_names.tolist())
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "hierarchical_dendrogram.png")
    plt.close()


def plot_tsne(performance):
    """Apply t-SNE to visualize model relationships."""
    X = performance.values.T
    model_names = performance.columns

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])

    for i, name in enumerate(model_names):
        plt.text(X_tsne[i, 0], X_tsne[i, 1], name)

    plt.title("t-SNE Projection")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tsne_projection.png")
    plt.close()


def plot_extra(h2h, win):
    """Optional: visualize head-to-head and win counts."""
    if h2h is not None:
        plt.figure(figsize=(8, 6))
        sns.heatmap(h2h, annot=True, cmap="viridis")
        plt.title("Head-to-Head Wins")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "model_head_to_head_heatmap.png")
        plt.close()

    if win is not None:
        plt.figure(figsize=(8, 5))
        win["wins"].plot(kind="bar")
        plt.title("Model Win Counts")
        plt.ylabel("Wins")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "model_win_counts.png")
        plt.close()


def main():
    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        print("Run previous steps first.")
        return

    print("=== Generating visualizations ===\n")

    corr, cosine, euclidean, performance, h2h, win = load_results()

    plot_heatmaps(corr, cosine, euclidean)
    plot_kmeans(performance)
    plot_dendrogram(performance)
    plot_tsne(performance)
    plot_extra(h2h, win)

    print("All visualizations saved to results/")


if __name__ == "__main__":
    main()
