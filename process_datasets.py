from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

MISSING_MARKERS = {"?", "", "na", "nan", "null", "n/a", "none"}


def list_dataset_dirs(parent_dir: Path) -> List[Path]:
    return sorted([p for p in parent_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower())


def first_matching_file(folder: Path, pattern: str) -> Optional[Path]:
    matches = sorted([p for p in folder.glob(pattern) if p.is_file()], key=lambda p: p.name.lower())
    return matches[0] if matches else None


def clean_attr_name(raw_name: str) -> str:
    name = raw_name.strip()
    if (name.startswith("'") and name.endswith("'")) or (name.startswith('"') and name.endswith('"')):
        return name[1:-1]
    return name


def classify_arff_type(type_part: str) -> str:
    decl = type_part.strip()
    decl_no_comment = re.split(r"\s*%.*$", decl, maxsplit=1)[0].strip()
    # Nominal ARFF declarations enumerate values with braces
    if "{" in decl_no_comment and "}" in decl_no_comment:
        return "categorical"
    if decl_no_comment.lower() in {"numeric", "real", "integer"}:
        return "continuous"
    return "categorical"


def parse_arff(arff_path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    attr_names: List[str] = []
    attr_types: Dict[str, str] = {}
    data_rows: List[List[str]] = []
    in_data = False
    with arff_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            low = line.lower()
            if not in_data:
                if low.startswith("@data"):
                    in_data = True
                    continue
                if low.startswith("@attribute"):
                    parts = line.split(None, 2)
                    if len(parts) < 3:
                        continue
                    name = clean_attr_name(parts[1])
                    tkn = parts[2].strip()
                    attr_names.append(name)
                    attr_types[name] = classify_arff_type(tkn)
                continue
            row = next(csv.reader([line]))
            if row:
                data_rows.append([cell.strip() for cell in row])

    df = pd.DataFrame(data_rows, columns=attr_names)
    for col in df.columns:
        df[col] = df[col].map(lambda x: np.nan if str(x).strip().lower() in MISSING_MARKERS else x)
    return df, attr_types


def load_r_dat(r_path: Path) -> pd.DataFrame:
    df = pd.read_csv(r_path, sep=r"\s+", header=0, engine="python", comment="%")
    if df.shape[1] > 1:
        first_col_name = str(df.columns[0]).strip().lower()
        first_col_vals = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        seq = pd.Series(np.arange(1, len(df) + 1), index=df.index, dtype=float)
        is_unnamed = first_col_name.startswith("unnamed")
        is_seq_index = first_col_vals.notna().all() and np.allclose(
            first_col_vals.to_numpy(dtype=float), seq.to_numpy(dtype=float)
        )
        if is_unnamed or is_seq_index:
            df = df.iloc[:, 1:].copy()
    return df


def build_version_b(arff_df: pd.DataFrame, attr_types: Dict[str, str]) -> Tuple[pd.DataFrame, int, int, int, int, int, float]:
    target_col = arff_df.columns[-1]
    feature_cols = list(arff_df.columns[:-1])
    target = arff_df[target_col]
    categorical = [c for c in feature_cols if attr_types.get(c, "categorical") == "categorical"]
    continuous = [c for c in feature_cols if c not in categorical]
    X = arff_df[feature_cols].copy()
    missing_count = int(X.isna().sum().sum() + target.isna().sum())

    for col in continuous:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())
    for col in categorical:
        mode = X[col].mode(dropna=True)
        X[col] = X[col].fillna(mode.iloc[0] if not mode.empty else "missing").astype(str)

    X_cont = pd.DataFrame(index=X.index)
    if continuous:
        X_cont = pd.DataFrame(StandardScaler().fit_transform(X[continuous]), columns=continuous, index=X.index)
    X_cat = pd.DataFrame(index=X.index)
    if categorical:
        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        X_cat = pd.DataFrame(enc.fit_transform(X[categorical]), columns=enc.get_feature_names_out(categorical), index=X.index)

    X_final = pd.concat([X_cont, X_cat], axis=1)
    version_b = pd.concat([X_final, target.rename(target_col)], axis=1)
    target_no_na = target.dropna()
    maj_pct = float(target_no_na.value_counts(normalize=True).max() * 100) if not target_no_na.empty else 0.0
    return version_b, len(feature_cols), X_final.shape[1], len(categorical), len(continuous), missing_count, maj_pct


def build_version_c(arff_df: pd.DataFrame, attr_types: Dict[str, str]) -> Tuple[pd.DataFrame, int, int, int, int, int, float]:
    target_col = arff_df.columns[-1]
    feature_cols = list(arff_df.columns[:-1])
    target = arff_df[target_col]
    categorical = [c for c in feature_cols if attr_types.get(c, "categorical") == "categorical"]
    continuous = [c for c in feature_cols if c not in categorical]
    X = arff_df[feature_cols].copy()
    missing_count = int(X.isna().sum().sum() + target.isna().sum())

    for col in continuous:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())
    for col in categorical:
        mode = X[col].mode(dropna=True)
        X[col] = X[col].fillna(mode.iloc[0] if not mode.empty else "missing").astype(str)

    X_cont = pd.DataFrame(index=X.index)
    if continuous:
        X_cont = pd.DataFrame(MinMaxScaler().fit_transform(X[continuous]), columns=continuous, index=X.index)
    X_cat = pd.DataFrame(index=X.index)
    if categorical:
        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        X_cat = pd.DataFrame(enc.fit_transform(X[categorical]), columns=enc.get_feature_names_out(categorical), index=X.index)

    X_final = pd.concat([X_cont, X_cat], axis=1)
    version_c = pd.concat([X_final, target.rename(target_col)], axis=1)
    target_no_na = target.dropna()
    maj_pct = float(target_no_na.value_counts(normalize=True).max() * 100) if not target_no_na.empty else 0.0
    return version_c, len(feature_cols), X_final.shape[1], len(categorical), len(continuous), missing_count, maj_pct


def process_dataset(folder: Path, idx: int, total: int) -> Tuple[Dict[str, object], List[str]]:
    name = folder.name
    notes: List[str] = []
    has_a = False
    has_b = False
    has_c = False
    r_file = first_matching_file(folder, "*_R.dat")
    arff_file = first_matching_file(folder, "*.arff")

    if r_file is None:
        notes.append("Version A skipped: no _R.dat")
    else:
        try:
            load_r_dat(r_file).to_csv(folder / f"{name}_version_a.csv", index=False)
            has_a = True
        except Exception as exc:
            notes.append(f"Version A failed: {exc}")

    num_samples = np.nan
    num_features_original = np.nan
    num_features_version_b = np.nan
    num_features_version_c = np.nan
    num_classes = np.nan
    num_categorical = np.nan
    num_continuous = np.nan
    num_missing = np.nan
    majority_pct = np.nan

    if arff_file is None:
        notes.append("Version B skipped: no .arff")
    else:
        try:
            arff_df, attr_types = parse_arff(arff_file)
            df_b, n_orig, n_b, n_cat, n_cont, n_miss, maj_pct = build_version_b(arff_df, attr_types)
            df_b.to_csv(folder / f"{name}_version_b.csv", index=False)
            has_b = True
            df_c, _, n_c, _, _, _, _ = build_version_c(arff_df, attr_types)
            df_c.to_csv(folder / f"{name}_version_c.csv", index=False)
            has_c = True
            num_samples = len(arff_df)
            num_features_original = n_orig
            num_features_version_b = n_b
            num_features_version_c = n_c
            num_classes = int(arff_df.iloc[:, -1].dropna().nunique()) if not arff_df.empty else 0
            num_categorical = n_cat
            num_continuous = n_cont
            num_missing = n_miss
            majority_pct = maj_pct
        except Exception as exc:
            notes.append(f"Version B failed: {exc}")

    status = "done" if (has_a or has_b or has_c) else "SKIPPED"
    if notes:
        print(f"Processing {idx}/{total}: {name}... {status} ({'; '.join(notes)})")
    else:
        print(f"Processing {idx}/{total}: {name}... {status}")

    return {
        "dataset_name": name,
        "num_samples": num_samples,
        "num_features_original": num_features_original,
        "num_features_version_b": num_features_version_b,
        "num_features_version_c": num_features_version_c,
        "num_classes": num_classes,
        "num_categorical_features": num_categorical,
        "num_continuous_features": num_continuous,
        "num_missing_values": num_missing,
        "majority_class_percentage": majority_pct,
        "has_version_a": has_a,
        "has_version_b": has_b,
        "has_version_c": has_c,
        "notes": "; ".join(notes),
    }, notes


def main(parent_dir: Path) -> None:
    datasets = list_dataset_dirs(parent_dir)
    total = len(datasets)
    rows: List[Dict[str, object]] = []
    issues: List[Tuple[str, List[str]]] = []

    for i, folder in enumerate(datasets, 1):
        row, notes = process_dataset(folder, i, total)
        rows.append(row)
        if notes:
            issues.append((folder.name, notes))

    out = parent_dir / "dataset_metadata.csv"
    mdf = pd.DataFrame(rows)
    mdf.to_csv(out, index=False)

    both = sum(1 for r in rows if r["has_version_a"] and r["has_version_b"])
    any_success = sum(1 for r in rows if r["has_version_a"] or r["has_version_b"])

    reason_counts: Dict[str, int] = {}
    for _, note_list in issues:
        for n in note_list:
            reason_counts[n] = reason_counts.get(n, 0) + 1

    print("\n--- Summary ---")
    print(f"Total datasets processed: {total}")
    print(f"Succeeded (both A and B): {both}")
    print(f"At least one version produced: {any_success}")
    print(f"Skipped or failed: {total - both}")
    print(f"Datasets with issues/skips: {len(issues)}")
    if reason_counts:
        print("Failure/skip reasons:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  - {count}x {reason}")
    print(f"\nMetadata saved to: {out}")


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "data-20260323T043051Z-3-001" / "data"
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Usage: place dataset folders under data-20260323T043051Z-3-001/data/")
    else:
        main(data_dir)
