# DS 2500 Spring — ML Classifier Similarity Analysis

## Project Overview

This project investigates how similarly different machine learning classifiers behave across 121 UCI datasets. Rather than just comparing raw accuracy, we analyze the *behavioral fingerprints* of classifiers — where they succeed and fail together — to discover which models "think alike" regardless of their algorithmic family.

Based on the methodology of Fernández-Delgado et al. *"Do we Need Hundreds of Classifiers to Solve Real World Classification Problems?"*

---

## Team & Contributions

| Member | Role |
|--------|------|
| **Daniel Rodas Sanchez** | Data collection, cleaning, and preprocessing (`process_datasets.py`) |
| **Varun Khedkar** | Similarity computation between models — correlation, cosine similarity, Euclidean distance (`similarity_analysis.py`) |
| **Daniel Si-Hyun Ryu** | Clustering analysis — K-Means and hierarchical clustering to group similar models |
| **Zhiheng Shao** | Visualization — heatmaps, t-SNE/UMAP plots, presentation slides |

---

## Repo Structure

```
DS_Spring_2500_ML/
├── data-20260323T043051Z-3-001/data/   # 121 UCI dataset folders (tracked via Git LFS)
├── notebooks/
│   └── similarity_analysis_colab.ipynb # Colab notebook for running full analysis
├── results/                            # Output CSVs (generated, not committed)
│   ├── dataset_metadata.csv
│   ├── model_performance_matrix.csv
│   ├── model_correlation_matrix.csv
│   ├── model_cosine_similarity_matrix.csv
│   └── model_euclidean_distance_matrix.csv
├── process_datasets.py                 # Step 1: Preprocess raw datasets into versions A/B/C
├── similarity_analysis.py              # Step 2: Run classifiers + compute similarity matrices
├── requirements.txt
└── README.md
```

---

## Pipeline

### Step 1 — Preprocess Datasets (Daniel R.S.)

```bash
python process_datasets.py
```

Reads each dataset's `.arff` and `_R.dat` files and outputs three CSV versions per dataset:

| Version | Source | Preprocessing |
|---------|--------|---------------|
| **A** | `_R.dat` file | Raw data converted to CSV (no transformations) |
| **B** | `.arff` file | Missing values imputed (median for continuous, mode for categorical), continuous features scaled with **StandardScaler** (zero mean, unit variance), categorical features one-hot encoded |
| **C** | `.arff` file | Same imputation and encoding as B, but continuous features scaled with **MinMaxScaler** (0–1 range) |

For each dataset folder, the script generates:
- `<name>_version_a.csv` — raw version
- `<name>_version_b.csv` — StandardScaler version
- `<name>_version_c.csv` — MinMaxScaler version

Also outputs `results/dataset_metadata.csv` with per-dataset statistics:
- Number of samples, original features, transformed features
- Number of categorical vs. continuous features
- Missing value counts
- Majority class percentage
- Which versions were successfully produced

By default, the script looks for datasets in `data-20260323T043051Z-3-001/data/` relative to the script location. Each subdirectory should contain an `.arff` file and/or a `_R.dat` file.

### Step 2 — Similarity Analysis (Varun) ✅ Complete

```bash
python similarity_analysis.py
```

Runs 8 classifiers on every dataset (up to 10k rows) using 5-fold cross-validation, then computes three pairwise similarity matrices across all models:

- **Pearson Correlation** — do two models struggle/succeed on the same datasets?
- **Cosine Similarity** — do two models have the same performance profile shape?
- **Euclidean Distance** — how far apart are two models' raw accuracy vectors?

**Results (102 datasets, 8 classifiers) — already in `results/`:**
- `model_performance_matrix.csv` — raw accuracy per model per dataset
- `model_correlation_matrix.csv` — key finding: SVM↔NeuralNet (0.96), SVM↔RandomForest (0.95), AdaBoost↔NaiveBayes lowest (0.58)
- `model_cosine_similarity_matrix.csv`
- `model_euclidean_distance_matrix.csv`
- `model_win_counts.csv` — how many datasets each model wins outright
- `model_head_to_head.csv` — pairwise dominance: how often model A beats model B

> **Side finding — rank analysis:** High correlation ≠ equal performance. SVM and RandomForest correlate at 0.95 but SVM wins on 56/102 head-to-head matchups vs RandomForest's 41. LogReg dominates NaiveBayes on 91/102 datasets despite both being "linear" models. Top winners: RandomForest (26 datasets), SVM (22), NeuralNet (18), LogReg (14).

### Step 3 — Clustering (Daniel Ryu) ← Your turn

**Inputs ready in `results/`** — pull and use these directly:
- `model_correlation_matrix.csv` — recommended primary input for clustering
- `model_cosine_similarity_matrix.csv` — alternative input
- `model_performance_matrix.csv` — raw accuracies if needed

**Tasks:**
- Apply K-Means clustering on the correlation matrix to group similar classifiers
- Apply hierarchical clustering and generate a dendrogram
- Validate cluster quality with silhouette scores
- Key question to answer: do classifiers cluster by algorithm family (e.g. tree-based vs linear) or by behavior?

**To get started:**
```bash
git pull origin main
# results/model_correlation_matrix.csv is your main input
```

### Step 4 — Visualization (Zhiheng)
Generates visual representations of model relationships and clustering results based on the computed similarity matrices and performance data:

1, Heatmaps — visualize pairwise relationships between models using:
Pearson correlation matrix
Cosine similarity matrix
Euclidean distance matrix
2, Clustering Visualizations — display grouping structure of models:
PCA-based scatter plots for K-Means clustering results
Hierarchical clustering dendrograms
3, Dimensionality Reduction — project high-dimensional model performance into 2D space:
t-SNE or UMAP plots to reveal patterns and similarities among models

Outputs to results/:
model_correlation_heatmap.png
model_cosine_heatmap.png
model_euclidean_heatmap.png
kmeans_clusters.png
hierarchical_dendrogram.png
tsne_projection.png


## Running on Google Colab

For the full 121-dataset run, use the Colab notebook to avoid tying up your local machine:

1. Open [Google Colab](https://colab.research.google.com)
2. File → Upload notebook → select `notebooks/similarity_analysis_colab.ipynb`
3. Run all cells — the last cell downloads `similarity_results.zip` automatically

---

## Requirements

```bash
pip install -r requirements.txt
```

- Python 3.8+
- pandas
- numpy
- scikit-learn
- Git LFS (for dataset files — `brew install git-lfs && git lfs pull`)
