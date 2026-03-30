import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

CLASSIFIERS = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "SVM": SVC(kernel="rbf", random_state=42),
    "NeuralNet": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "NaiveBayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "LogReg": LogisticRegression(max_iter=1000, random_state=42),
}

DATA_DIR = Path(__file__).resolve().parent / "data-20260323T043051Z-3-001" / "data"
OUT_DIR  = Path(__file__).resolve().parent / "results"


def load_dataset(version_b_path):
    """Load a version_b CSV and split into features (X) and encoded target (y)."""
    df = pd.read_csv(version_b_path)
    X = df.iloc[:, :-1].values
    y = LabelEncoder().fit_transform(df.iloc[:, -1].astype(str))
    return X, y


def build_performance_matrix(data_dir, classifiers, cv_folds=5):
    """
    Run each classifier on every dataset using cross-validation
    and return a DataFrame of accuracies (rows=datasets, cols=models).
    """
    dataset_dirs = sorted(
        [p for p in data_dir.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower(),
    )

    results = {}
    total = len(dataset_dirs)

    for i, folder in enumerate(dataset_dirs, 1):
        name = folder.name
        version_b = folder / f"{name}_version_b.csv"
        if not version_b.exists():
            print(f"[{i}/{total}] {name}: version_b not found, skipping")
            continue

        try:
            X, y = load_dataset(version_b)
        except Exception as e:
            print(f"[{i}/{total}] {name}: failed to load ({e}), skipping")
            continue



        # Need at least cv_folds samples per class
        unique, counts = np.unique(y, return_counts=True)
        if counts.min() < cv_folds:
            print(f"[{i}/{total}] {name}: too few samples in a class, skipping")
            continue

        # For large datasets, swap in faster classifiers
        if X.shape[0] > 5000:
            run_clfs = {k: v for k, v in classifiers.items()}
            run_clfs["SVM"] = LinearSVC(max_iter=2000, random_state=42)
            run_clfs["KNN"] = KNeighborsClassifier(n_neighbors=5, algorithm="ball_tree")
        else:
            run_clfs = classifiers

        row = {}
        for clf_name, clf in run_clfs.items():
            try:
                scores = cross_val_score(clf, X, y, cv=cv_folds, scoring="accuracy")
                row[clf_name] = scores.mean()
            except Exception:
                row[clf_name] = np.nan

        results[name] = row
        print(f"[{i}/{total}] {name}: done")

    df = pd.DataFrame.from_dict(results, orient="index")
    df = df.dropna(axis=0, how="any")
    return df


def compute_model_similarities(df_performance):
    """
    Computes similarity matrices for machine learning models based on
    their performance across multiple datasets.

    Returns a dict with correlation, cosine_similarity, and euclidean_distance matrices.
    """
    corr_matrix = df_performance.corr(method="pearson")

    cos_sim_array = cosine_similarity(df_performance.T)
    cos_sim_matrix = pd.DataFrame(
        cos_sim_array,
        index=df_performance.columns,
        columns=df_performance.columns,
    )

    euc_dist_array = euclidean_distances(df_performance.T)
    euc_dist_matrix = pd.DataFrame(
        euc_dist_array,
        index=df_performance.columns,
        columns=df_performance.columns,
    )

    return {
        "correlation": corr_matrix,
        "cosine_similarity": cos_sim_matrix,
        "euclidean_distance": euc_dist_matrix,
    }


if __name__ == "__main__":
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        print("Run process_datasets.py first to generate version_b CSVs.")
        exit(1)

    OUT_DIR.mkdir(exist_ok=True)

    # Step 1: Build performance matrix by running classifiers on real datasets
    print("=== Building performance matrix (this may take a while) ===\n")
    df_performance = build_performance_matrix(DATA_DIR, CLASSIFIERS)
    df_performance.to_csv(OUT_DIR / "model_performance_matrix.csv")
    print(f"\nPerformance matrix: {df_performance.shape[0]} datasets x {df_performance.shape[1]} models")
    print(f"Saved results/model_performance_matrix.csv\n")

    # Step 2: Compute similarity metrics
    print("=== Computing similarity metrics ===\n")
    similarities = compute_model_similarities(df_performance)

    similarities["correlation"].to_csv(OUT_DIR / "model_correlation_matrix.csv")
    similarities["cosine_similarity"].to_csv(OUT_DIR / "model_cosine_similarity_matrix.csv")
    similarities["euclidean_distance"].to_csv(OUT_DIR / "model_euclidean_distance_matrix.csv")

    print("Saved to results/:")
    print("  - model_performance_matrix.csv (raw accuracies)")
    print("  - model_correlation_matrix.csv")
    print("  - model_cosine_similarity_matrix.csv")
    print("  - model_euclidean_distance_matrix.csv")
    print("\nThese matrices are ready for clustering and visualization.")
