# DS 2500 Spring — ML Dataset Preprocessing Pipeline

## Overview

This project preprocesses 124 UCI-style machine learning datasets into three standardized versions, making them ready for downstream classification experiments. Each dataset is converted from its original format (ARFF / R `.dat`) into clean CSV files with consistent preprocessing applied.

## Dataset Versions

| Version | Source | Preprocessing |
|---------|--------|---------------|
| **A** | `_R.dat` file | Raw data converted to CSV (no transformations) |
| **B** | `.arff` file | Missing values imputed (median for continuous, mode for categorical), continuous features scaled with **StandardScaler** (zero mean, unit variance), categorical features one-hot encoded |
| **C** | `.arff` file | Same imputation and encoding as B, but continuous features scaled with **MinMaxScaler** (0-1 range) |

## Output

For each dataset folder, the script generates:
- `<name>_version_a.csv` — raw version
- `<name>_version_b.csv` — StandardScaler version
- `<name>_version_c.csv` — MinMaxScaler version

A summary file `dataset_metadata.csv` is saved to the parent data directory with per-dataset statistics:
- Number of samples, original features, transformed features
- Number of categorical vs. continuous features
- Missing value counts
- Majority class percentage
- Which versions were successfully produced

## Usage

```bash
python process_datasets.py
```

By default, the script looks for datasets in `data-20260323T043051Z-3-001/data/` relative to the script location. Each subdirectory should contain an `.arff` file and/or a `_R.dat` file.

## Requirements

- Python 3.14+
- pandas
- numpy
- scikit-learn
