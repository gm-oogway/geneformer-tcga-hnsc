"""Step 02: merge per-sample STAR-Counts files into one samples x genes matrix.

Input : data/raw/<file_id>/<file_name>.tsv   (one file per sample, from step 01)
        metadata/gdc_sample_sheet*.tsv        (maps file -> patient / sample / tissue type)
Output: data/interim/hnsc_counts.h5ad         (AnnData: raw counts + sample and gene annotation)

Usage : python scripts/02_build_matrix.py            # all samples
        python scripts/02_build_matrix.py --limit 5  # quick test on 5 samples
"""
import argparse
import glob
import os

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None, help="only use the first N samples (for testing)")
parser.add_argument("--out", default="data/interim/hnsc_counts.h5ad")
args = parser.parse_args()

# 1. Sample sheet: one row per downloaded file
sheet = pd.read_csv(glob.glob("metadata/gdc_sample_sheet*.tsv")[0], sep="\t")
if args.limit:
    sheet = sheet.head(args.limit)
print(f"Samples to read: {len(sheet)}")

# 2. Read the raw-count column of every file
counts = []
genes = None
for i, row in enumerate(sheet.itertuples(index=False)):
    path = os.path.join("data/raw", row[0], row[1])  # File ID, File Name
    df = pd.read_csv(path, sep="\t", comment="#")     # skips the '# gene-model' line
    df = df[~df["gene_id"].str.startswith("N_")]      # drop the 4 STAR summary rows
    if genes is None:
        genes = df[["gene_id", "gene_name", "gene_type"]].reset_index(drop=True)
    else:
        # every file must list the same genes in the same order
        assert (df["gene_id"].values == genes["gene_id"].values).all(), f"gene order differs in {path}"
    counts.append(df["unstranded"].to_numpy(dtype=np.int32))
    if (i + 1) % 100 == 0:
        print(f"  read {i + 1}/{len(sheet)}")

X = np.vstack(counts)  # rows = samples, columns = genes
print(f"Raw matrix: {X.shape[0]} samples x {X.shape[1]} genes")

# 3. Clean gene IDs: ENSG00000000003.15 -> ENSG00000000003
keep = ~genes["gene_id"].str.contains("_PAR_Y").values  # duplicate copies of X/Y shared genes
X = X[:, keep]
genes = genes[keep].reset_index(drop=True)
genes["ensembl_id"] = genes["gene_id"].str.split(".").str[0]
assert genes["ensembl_id"].is_unique, "duplicate Ensembl IDs after removing versions"
print(f"Removed {int((~keep).sum())} PAR_Y genes -> {X.shape[1]} genes")

# 4. Sample annotation (obs)
obs = pd.DataFrame(
    {
        "file_id": sheet["File ID"].values,
        "case_id": sheet["Case ID"].values,
        "sample_id": sheet["Sample ID"].values,
        "tissue_type": sheet["Tissue Type"].values,
        "tumor_descriptor": sheet["Tumor Descriptor"].values,
        "specimen_type": sheet["Specimen Type"].values,
    }
)
obs["n_counts"] = X.sum(axis=1).astype(np.int64)  # total reads per sample (Geneformer needs this)
obs.index = obs["sample_id"].astype(str).values

# 5. Gene annotation (var)
var = genes[["ensembl_id", "gene_name", "gene_type"]].copy()
var.index = var["ensembl_id"].values

adata = ad.AnnData(X=sparse.csr_matrix(X), obs=obs, var=var)
adata.obs_names_make_unique()  # two files from the same sample get suffixes -1, -2

# 6. Report and save
print(adata)
print("\nTissue type:\n", adata.obs["tissue_type"].value_counts().to_string())
print("\nPatients:", adata.obs["case_id"].nunique())
print("Samples sharing a sample_id:", int(adata.obs["sample_id"].duplicated().sum()))
print("Library size (million reads): min %.1f, median %.1f, max %.1f"
      % tuple(np.percentile(adata.obs["n_counts"] / 1e6, [0, 50, 100])))
os.makedirs(os.path.dirname(args.out), exist_ok=True)
adata.write_h5ad(args.out)
print(f"\nSaved {args.out}")
