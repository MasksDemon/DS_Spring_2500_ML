"""
Clustering Analysis of ML Classifiers
======================================
Primary input : results/model_correlation_matrix.csv
Alternative   : results/model_cosine_similarity_matrix.csv
Fallback      : results/model_performance_matrix.csv (raw accuracies)

Tasks
-----
1. K-Means clustering on the correlation matrix to group similar classifiers.
2. Hierarchical clustering + dendrogram (using 1 - correlation as distance).
3. Silhouette score validation for cluster quality.
4. Key question: do classifiers cluster by algorithm family
   (tree-based vs linear) or by behaviour across datasets?
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_DIR            = Path(__file__).resolve().parent / "results"
CORR_CSV           = OUT_DIR / "model_correlation_matrix.csv"
COSINE_CSV         = OUT_DIR / "model_cosine_similarity_matrix.csv"
PERFORMANCE_CSV    = OUT_DIR / "model_performance_matrix.csv"

# ---------------------------------------------------------------------------
# Algorithm-family labels (for answering the key research question)
# ---------------------------------------------------------------------------
ALGORITHM_FAMILY = {
    "RandomForest":  "Ensemble (Tree)",
    "AdaBoost":      "Ensemble (Tree)",
    "DecisionTree":  "Tree",
    "SVM":           "Kernel / Linear",
    "LogReg":        "Kernel / Linear",
    "KNN":           "Instance-based",
    "NaiveBayes":    "Probabilistic",
    "NeuralNet":     "Neural Network",
}

FAMILY_COLORS = {
    "Ensemble (Tree)":  "#2ca02c",
    "Tree":             "#98df8a",
    "Kernel / Linear":  "#1f77b4",
    "Instance-based":   "#ff7f0e",
    "Probabilistic":    "#9467bd",
    "Neural Network":   "#d62728",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_similarity_matrix(primary: Path, fallback_cosine: Path,
                            fallback_perf: Path) -> tuple[pd.DataFrame, str]:
    """
    Load the model similarity matrix.

    Priority:
      1. Correlation matrix  (primary — recommended)
      2. Cosine-similarity matrix (alternative)
      3. Performance matrix transposed  (last resort)

    Returns:
        (square DataFrame indexed/columned by model names, source description)
    """
    if primary.exists():
        df = pd.read_csv(primary, index_col=0)
        return df, "correlation matrix"
    if fallback_cosine.exists():
        df = pd.read_csv(fallback_cosine, index_col=0)
        return df, "cosine-similarity matrix"
    if fallback_perf.exists():
        df = pd.read_csv(fallback_perf, index_col=0).dropna().T
        corr = df.T.corr(method="pearson")
        return corr, "correlation derived from performance matrix"
    raise FileNotFoundError(
        "No input matrix found. Run similarity_analysis.py first."
    )


# ---------------------------------------------------------------------------
# K-Means on correlation matrix
# ---------------------------------------------------------------------------

def kmeans_on_correlation(sim_matrix: pd.DataFrame, max_k: int = 6):
    """
    Apply K-Means clustering using the rows of the correlation matrix as
    feature vectors.  Each model is represented by its correlation with
    every other model.

    Steps:
      1. Use correlation rows directly as features (already normalised).
      2. Sweep k=2..max_k, compute inertia + silhouette score.
      3. Select best k by highest silhouette score.

    Returns:
        labels     : dict  model -> cluster id (0-indexed)
        best_k     : int
        k_range    : list[int]
        inertias   : list[float]
        sil_scores : list[float]
    """
    X = sim_matrix.values          # shape: (n_models, n_models)
    models = list(sim_matrix.index)
    n = len(models)
    max_k = min(max_k, n - 1)

    k_range    = list(range(2, max_k + 1))
    inertias   = []
    sil_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        km.fit(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, km.labels_))

    best_k = k_range[int(np.argmax(sil_scores))]

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    km_final.fit(X)
    labels = {m: int(l) for m, l in zip(models, km_final.labels_)}

    return labels, best_k, k_range, inertias, sil_scores


# ---------------------------------------------------------------------------
# Hierarchical clustering using 1 - correlation as distance
# ---------------------------------------------------------------------------

def hierarchical_on_correlation(sim_matrix: pd.DataFrame, method: str = "average"):
    """
    Build a linkage matrix using (1 - correlation) as the pairwise distance.

    'average' linkage is the standard choice for correlation-based distances
    (also called UPGMA).  'ward' is omitted here because it requires
    Euclidean distances, not correlation distances.

    Returns:
        Z      : linkage matrix
        models : list of model names (preserves sim_matrix.index order)
    """
    corr = sim_matrix.values.copy()
    np.clip(corr, -1, 1, out=corr)
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)
    return Z, list(sim_matrix.index)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_elbow_silhouette(k_range, inertias, sil_scores, best_k, out_path: Path):
    """Elbow + silhouette plots side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(k_range, inertias, marker="o", color="steelblue", linewidth=2)
    axes[0].axvline(best_k, color="red", linestyle="--", linewidth=1,
                    label=f"Best k = {best_k}")
    axes[0].set_title("Elbow Method — K-Means Inertia")
    axes[0].set_xlabel("Number of Clusters k")
    axes[0].set_ylabel("Inertia")
    axes[0].set_xticks(k_range)
    axes[0].legend()

    axes[1].plot(k_range, sil_scores, marker="o", color="darkorange", linewidth=2)
    axes[1].axvline(best_k, color="red", linestyle="--", linewidth=1,
                    label=f"Best k = {best_k}")
    axes[1].set_title("Silhouette Score vs k (K-Means)")
    axes[1].set_xlabel("Number of Clusters k")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_xticks(k_range)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path.name}")


def plot_kmeans_assignment(labels: dict, best_k: int, out_path: Path):
    """
    Bar chart: classifier vs cluster, coloured by algorithm family.
    Answers the key question visually.
    """
    models   = list(labels.keys())
    clusters = [labels[m] + 1 for m in models]   # 1-indexed for display
    families = [ALGORITHM_FAMILY.get(m, "Other") for m in models]
    colors   = [FAMILY_COLORS.get(f, "#7f7f7f") for f in families]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(models, clusters, color=colors, edgecolor="black", width=0.55)

    ax.set_title(f"K-Means Cluster Assignments (k={best_k})\n"
                 "Bar colour = algorithm family")
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Cluster")
    ax.set_yticks(range(1, best_k + 1))
    ax.set_yticklabels([f"Cluster {i}" for i in range(1, best_k + 1)])
    plt.xticks(rotation=30, ha="right")

    # Legend for algorithm families
    seen = {}
    for m, f in zip(models, families):
        seen[f] = FAMILY_COLORS.get(f, "#7f7f7f")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=f)
               for f, c in seen.items()]
    ax.legend(handles=handles, title="Algorithm Family",
              loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path.name}")


def plot_dendrogram(Z, models, best_k: int, out_path: Path, method: str = "average"):
    """
    Dendrogram coloured by algorithm family (leaf labels).
    A horizontal dashed line shows the cut that produces best_k clusters.
    """
    # Determine colour threshold for the dendrogram colouring
    n_merges = len(Z)
    cut_idx  = n_merges - (best_k - 1)          # merge index where we get best_k clusters
    if 0 < cut_idx <= n_merges:
        color_threshold = Z[cut_idx - 1, 2] * 1.01
    else:
        color_threshold = Z[-1, 2] * 0.7

    fig, ax = plt.subplots(figsize=(10, 5))
    ddata = dendrogram(
        Z,
        labels=models,
        color_threshold=color_threshold,
        ax=ax,
        leaf_rotation=35,
        leaf_font_size=11,
    )
    ax.axhline(y=color_threshold, color="grey", linestyle="--", linewidth=1,
               label=f"Cut → {best_k} clusters")

    # Colour the leaf labels by algorithm family
    xlabels = ax.get_xticklabels()
    for lbl in xlabels:
        model  = lbl.get_text()
        family = ALGORITHM_FAMILY.get(model, "Other")
        lbl.set_color(FAMILY_COLORS.get(family, "#333333"))
        lbl.set_fontweight("bold")

    ax.set_title(f"Hierarchical Clustering Dendrogram ({method} linkage, "
                 f"distance = 1 − correlation)\nLeaf colour = algorithm family")
    ax.set_xlabel("Classifier")
    ax.set_ylabel("1 − Pearson Correlation (distance)")
    ax.legend(fontsize=9)

    # Add a compact family colour legend
    seen = {ALGORITHM_FAMILY.get(m, "Other"): FAMILY_COLORS.get(
                ALGORITHM_FAMILY.get(m, "Other"), "#333333")
            for m in models}
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=f)
               for f, c in seen.items()]
    ax.legend(handles=handles, title="Algorithm Family",
              loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path.name}")


def plot_silhouette_samples(sim_matrix: pd.DataFrame, labels: dict,
                            best_k: int, out_path: Path):
    """Per-classifier silhouette scores — shows which models are well-separated."""
    X       = sim_matrix.values
    models  = list(sim_matrix.index)
    y       = np.array([labels[m] for m in models])
    s_vals  = silhouette_samples(X, y)

    families = [ALGORITHM_FAMILY.get(m, "Other") for m in models]
    colors   = [FAMILY_COLORS.get(f, "#7f7f7f") for f in families]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(models, s_vals, color=colors, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(float(np.mean(s_vals)), color="red", linestyle="--", linewidth=1,
               label=f"Mean = {np.mean(s_vals):.3f}")
    ax.set_title(f"Per-Classifier Silhouette Scores (k={best_k})\n"
                 "Bar colour = algorithm family")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Classifier")
    ax.legend(fontsize=9)

    seen = {}
    for m, f in zip(models, families):
        seen[f] = FAMILY_COLORS.get(f, "#7f7f7f")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=f)
               for f, c in seen.items()]
    ax.legend(handles=handles + [
        plt.Line2D([0], [0], color="red", linestyle="--",
                   label=f"Mean = {np.mean(s_vals):.3f}")
    ], title="Algorithm Family", loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Key-question analysis
# ---------------------------------------------------------------------------

def analyse_family_vs_cluster(kmeans_labels: dict, hier_labels: dict) -> str:
    """
    Compare cluster assignments against known algorithm families.
    Returns a short textual summary for stdout and the CSV.
    """
    rows = []
    for model in kmeans_labels:
        rows.append({
            "model":                model,
            "algorithm_family":     ALGORITHM_FAMILY.get(model, "Other"),
            "kmeans_cluster":       kmeans_labels[model] + 1,
            "hierarchical_cluster": hier_labels.get(model, "?"),
        })
    df = pd.DataFrame(rows)

    lines = ["\n--- Key Question: Algorithm Family vs Cluster ---"]
    for method in ("kmeans_cluster", "hierarchical_cluster"):
        lines.append(f"\n  [{method}]")
        for cluster_id, group in df.groupby(method):
            families = group["algorithm_family"].tolist()
            models   = group["model"].tolist()
            unique_f = set(families)
            same_fam = len(unique_f) == 1
            tag = "(same family)" if same_fam else "(mixed families)"
            lines.append(f"    Cluster {cluster_id} {tag}:")
            for m, f in zip(models, families):
                lines.append(f"      {m:15s} [{f}]")

    lines.append("\n  Interpretation:")
    # Check whether tree-based classifiers co-cluster
    tree_models   = {m for m, f in ALGORITHM_FAMILY.items() if "Tree" in f or "Ensemble" in f}
    linear_models = {m for m, f in ALGORITHM_FAMILY.items() if "Linear" in f}

    km_tree_clusters  = {kmeans_labels[m] for m in tree_models  if m in kmeans_labels}
    km_linear_clusters = {kmeans_labels[m] for m in linear_models if m in kmeans_labels}

    if len(km_tree_clusters) == 1:
        lines.append("  * Tree-based models (RF, DT, AdaBoost) share ONE K-Means cluster → "
                     "they cluster by algorithm family.")
    else:
        lines.append("  * Tree-based models span multiple K-Means clusters → "
                     "behavioural similarity overrides family membership.")

    if len(km_linear_clusters) == 1:
        lines.append("  * Linear/kernel models (SVM, LogReg) share ONE K-Means cluster → "
                     "family grouping holds.")
    else:
        lines.append("  * Linear/kernel models span multiple clusters → "
                     "dataset-driven behaviour dominates.")

    return "\n".join(lines), df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    print("Loading similarity matrix...")
    sim_matrix, source = load_similarity_matrix(CORR_CSV, COSINE_CSV, PERFORMANCE_CSV)
    print(f"  Source : {source}")
    print(f"  Models : {list(sim_matrix.index)}")
    print(f"  Shape  : {sim_matrix.shape}")

    # ── K-Means on correlation matrix ─────────────────────────────────────
    print("\n=== K-Means Clustering (on correlation matrix) ===")
    kmeans_labels, best_k, k_range, inertias, sil_scores = kmeans_on_correlation(sim_matrix)

    print(f"  Best k (highest silhouette): {best_k}")
    for k, s in zip(k_range, sil_scores):
        marker = " ← best" if k == best_k else ""
        print(f"    k={k}  silhouette={s:.4f}{marker}")

    print("\n  Cluster assignments (K-Means):")
    cluster_groups: dict = {}
    for model, cid in sorted(kmeans_labels.items(), key=lambda x: x[1]):
        cluster_groups.setdefault(cid + 1, []).append(model)
    for cid in sorted(cluster_groups):
        print(f"    Cluster {cid}: {', '.join(cluster_groups[cid])}")

    plot_elbow_silhouette(k_range, inertias, sil_scores, best_k,
                          OUT_DIR / "kmeans_elbow_silhouette.png")
    plot_kmeans_assignment(kmeans_labels, best_k,
                           OUT_DIR / "kmeans_cluster_assignment.png")
    plot_silhouette_samples(sim_matrix, kmeans_labels, best_k,
                            OUT_DIR / "kmeans_silhouette_samples.png")

    # ── Hierarchical clustering (1 - correlation distance) ────────────────
    print("\n=== Hierarchical Clustering (distance = 1 − correlation, average linkage) ===")
    Z, models = hierarchical_on_correlation(sim_matrix, method="average")

    hier_raw    = fcluster(Z, t=best_k, criterion="maxclust")
    hier_labels = {m: int(l) for m, l in zip(models, hier_raw)}

    print(f"  Cutting dendrogram at k={best_k}:")
    hier_groups: dict = {}
    for model, cid in sorted(hier_labels.items(), key=lambda x: x[1]):
        hier_groups.setdefault(cid, []).append(model)
    for cid in sorted(hier_groups):
        print(f"    Cluster {cid}: {', '.join(hier_groups[cid])}")

    # Silhouette score for hierarchical result
    X_hier    = sim_matrix.values
    y_hier    = np.array([hier_labels[m] for m in models])
    hier_sil  = silhouette_score(X_hier, y_hier) if len(set(y_hier)) > 1 else float("nan")
    print(f"  Silhouette score (hierarchical, k={best_k}): {hier_sil:.4f}")

    plot_dendrogram(Z, models, best_k,
                    OUT_DIR / "hierarchical_dendrogram.png", method="average")

    # ── Key question: family vs behaviour ─────────────────────────────────
    summary_text, summary_df = analyse_family_vs_cluster(kmeans_labels, hier_labels)
    print(summary_text)

    # ── Save summary CSV ──────────────────────────────────────────────────
    summary_df.to_csv(OUT_DIR / "clustering_summary.csv", index=False)
    print(f"\n  Saved: clustering_summary.csv")

    print("\n=== All outputs written to results/ ===")
    for f in ["kmeans_elbow_silhouette.png", "kmeans_cluster_assignment.png",
              "kmeans_silhouette_samples.png", "hierarchical_dendrogram.png",
              "clustering_summary.csv"]:
        print(f"  results/{f}")
