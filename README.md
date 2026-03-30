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

| Version | Scaling | Encoding |
|---------|---------|----------|
| A | None (raw) | None |
| B | StandardScaler (zero mean, unit variance) | One-hot |
| C | MinMaxScaler (0–1 range) | One-hot |

Also outputs `results/dataset_metadata.csv` with per-dataset statistics.

### Step 2 — Similarity Analysis (Varun)
```bash
python similarity_analysis.py
```
Runs 8 classifiers on every dataset using 5-fold cross-validation, then computes three pairwise similarity matrices across all models:

- **Pearson Correlation** — do two models struggle/succeed on the same datasets?
- **Cosine Similarity** — do two models have the same performance profile shape?
- **Euclidean Distance** — how far apart are two models' raw accuracy vectors?

Outputs to `results/`:
- `model_performance_matrix.csv` — raw accuracy per model per dataset
- `model_correlation_matrix.csv`
- `model_cosine_similarity_matrix.csv`
- `model_euclidean_distance_matrix.csv`

### Step 3 — Clustering (Daniel Ryu)
Uses the similarity matrices from Step 2 to group classifiers with K-Means and hierarchical clustering. Validates cluster quality with silhouette scores.

### Step 4 — Visualization (Zhiheng)
Produces heatmaps and dimensionality reduction plots (t-SNE/UMAP) from the similarity and clustering results.

---

## Running on Google Colab

For the full 121-dataset run, use the Colab notebook to avoid tying up your local machine:

1. Open [Google Colab](https://colab.research.google.com)
2. File → Open notebook → GitHub tab → search `MasksDemon/DS_Spring_2500_ML`
3. Open `notebooks/similarity_analysis_colab.ipynb`
4. Run all cells — the last cell downloads `similarity_results.zip` automatically

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
