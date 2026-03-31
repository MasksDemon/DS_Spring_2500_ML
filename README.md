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
├── results/                            # Output CSVs
│   ├── dataset_metadata.csv
│   ├── model_performance_matrix.csv
│   ├── model_correlation_matrix.csv
│   ├── model_cosine_similarity_matrix.csv
│   ├── model_euclidean_distance_matrix.csv
│   ├── model_win_counts.csv
│   └── model_head_to_head.csv
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

Runs 8 classifiers on 102 datasets (≤10k rows) using 5-fold cross-validation. Treats each classifier's accuracy vector across datasets as a behavioral fingerprint, then computes three pairwise similarity metrics:

| Metric | What it captures |
|--------|-----------------|
| **Pearson Correlation** | Do two models succeed/fail on the same datasets? |
| **Cosine Similarity** | Do two models have the same performance profile shape? |
| **Euclidean Distance** | How far apart are two models' raw accuracy vectors? |

**Classifiers:** RandomForest, SVM, NeuralNet, DecisionTree, KNN, NaiveBayes, AdaBoost, LogReg

**Outputs in `results/`:**

| File | Description |
|------|-------------|
| `model_performance_matrix.csv` | Raw 5-fold CV accuracy — 102 datasets × 8 models |
| `model_correlation_matrix.csv` | Pearson correlation between model accuracy vectors |
| `model_cosine_similarity_matrix.csv` | Cosine similarity between model accuracy vectors |
| `model_euclidean_distance_matrix.csv` | Euclidean distance between model accuracy vectors |
| `model_win_counts.csv` | How many datasets each model wins outright |
| `model_head_to_head.csv` | Pairwise: how often model A beats model B across all datasets |

**Key findings:**
- SVM ↔ NeuralNet correlation: **0.96** — highest behavioral similarity
- SVM ↔ RandomForest correlation: **0.95** — nearly identical patterns
- AdaBoost ↔ NaiveBayes correlation: **0.58** — most behaviorally distinct pair
- Top dataset winners: RandomForest (26), SVM (22), NeuralNet (18), LogReg (14)
- High correlation ≠ equal dominance: SVM beats RandomForest head-to-head on 56/102 datasets despite 0.95 correlation

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

**Inputs ready in `results/`** — pull and use these directly:

- `model_correlation_matrix.csv` — primary input for similarity heatmaps
- `model_cosine_similarity_matrix.csv` — alternative similarity view
- `model_euclidean_distance_matrix.csv` — distance-based comparison
- `model_performance_matrix.csv` — raw accuracy data for projections and clustering visualizations

**Tasks:**

- Generate heatmaps to visualize relationships between models (correlation, cosine similarity, Euclidean distance)
- Visualize clustering results using PCA-based scatter plots (e.g. K-Means clusters)
- Generate hierarchical clustering dendrograms to show model grouping structure
- Apply dimensionality reduction (t-SNE or UMAP) to project model relationships into 2D space
- Compare visual patterns across methods to identify consistent groupings of models
- Key question to answer: do models group by algorithm family (e.g. tree-based vs linear) or by performance behavior?


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
