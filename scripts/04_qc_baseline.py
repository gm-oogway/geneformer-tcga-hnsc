"""Step 04: quality control and the classical baseline (normalise -> PCA -> UMAP).

Input : data/processed/hnsc_annotated.h5ad      (from step 03)
Output: data/processed/hnsc_baseline.h5ad       (normalised data + PCA + UMAP)
        results/tables/baseline_pca.csv         (first 20 principal components per sample)
        results/figures/04_*.png

Usage : python scripts/04_qc_baseline.py
"""
import os

import matplotlib

matplotlib.use("Agg")  # draw to files; WSL has no display window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

FIG = "results/figures"
os.makedirs(FIG, exist_ok=True)
N_HVG, N_PCS = 3000, 20

adata = sc.read_h5ad("data/processed/hnsc_annotated.h5ad")
adata.X = adata.X.astype(np.float32)
adata.layers["counts"] = adata.X.copy()  # keep raw counts untouched
print(adata)

# ---------- 1. Quality control ----------
adata.obs["n_genes_detected"] = np.asarray((adata.X > 0).sum(axis=1)).ravel()
adata.obs["library_size_M"] = adata.obs["n_counts"] / 1e6
print("\nQC summary:\n", adata.obs[["library_size_M", "n_genes_detected"]].describe().round(1).to_string())

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(adata.obs["library_size_M"], bins=40, color="steelblue")
ax[0].set(xlabel="Library size (million reads)", ylabel="Samples", title="Sequencing depth")
ax[1].hist(adata.obs["n_genes_detected"], bins=40, color="steelblue")
ax[1].set(xlabel="Genes with at least 1 read", ylabel="Samples", title="Genes detected")
for t, c in [("Tumor", "firebrick"), ("Normal", "seagreen")]:
    m = (adata.obs["tissue_type"] == t).values
    ax[2].scatter(adata.obs["library_size_M"][m], adata.obs["n_genes_detected"][m], s=10, c=c, label=t)
ax[2].set(xlabel="Library size (million reads)", ylabel="Genes detected", title="Depth vs genes detected")
ax[2].legend()
plt.tight_layout()
plt.savefig(f"{FIG}/04_qc.png", dpi=150)
plt.close()

# ---------- 2. Gene filtering ----------
# keep genes with at least 1 count-per-million in at least 10% of samples
cpm = adata.X.multiply(1e6 / adata.obs["n_counts"].values[:, None]).tocsr()
expressed = np.asarray((cpm > 1).sum(axis=0)).ravel() >= 0.10 * adata.n_obs
print(f"\nGenes kept after expression filter: {int(expressed.sum())} of {adata.n_vars}")
adata = adata[:, expressed].copy()

# ---------- 3. Normalisation ----------
sc.pp.normalize_total(adata, target_sum=1e6)  # counts per million: removes depth differences
sc.pp.log1p(adata)                            # log(1 + CPM): tames the huge dynamic range
adata.layers["logcpm"] = adata.X.copy()

# ---------- 4. Highly variable genes ----------
sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG)
print(f"Highly variable genes selected: {int(adata.var['highly_variable'].sum())}")

# ---------- 5. Scale + PCA (on the variable genes only) ----------
hv = adata[:, adata.var["highly_variable"]].copy()
sc.pp.scale(hv, max_value=10)                 # each gene: mean 0, variance 1
sc.tl.pca(hv, n_comps=50, random_state=0)
adata.obsm["X_pca"] = hv.obsm["X_pca"]
var_ratio = hv.uns["pca"]["variance_ratio"]
adata.uns["pca_variance_ratio"] = var_ratio
print("Variance explained by PC1-5 (%):", np.round(var_ratio[:5] * 100, 1))
print(f"Cumulative variance, first {N_PCS} PCs: {var_ratio[:N_PCS].sum() * 100:.1f}%")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(1, 51), var_ratio * 100, "o-", ms=3)
ax.axvline(N_PCS, color="grey", ls="--")
ax.set(xlabel="Principal component", ylabel="Variance explained (%)", title="PCA scree plot")
plt.tight_layout()
plt.savefig(f"{FIG}/04_pca_scree.png", dpi=150)
plt.close()

# ---------- 6. Neighbour graph + UMAP ----------
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=N_PCS, use_rep="X_pca", random_state=0)
sc.tl.umap(adata, random_state=0)

colors = ["tissue_type", "site", "stage", "sex", "smoking_status", "library_size_M"]
sc.pl.pca(adata, color=["tissue_type", "site"], show=False, ncols=2)
plt.savefig(f"{FIG}/04_pca.png", dpi=150, bbox_inches="tight")
plt.close()
sc.pl.umap(adata, color=colors, show=False, ncols=2, wspace=0.5)
plt.savefig(f"{FIG}/04_umap_baseline.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------- 7. Save ----------
pcs = pd.DataFrame(adata.obsm["X_pca"][:, :N_PCS], index=adata.obs_names,
                   columns=[f"PC{i + 1}" for i in range(N_PCS)])
os.makedirs("results/tables", exist_ok=True)
pcs.to_csv("results/tables/baseline_pca.csv")
adata.write_h5ad("data/processed/hnsc_baseline.h5ad")
print("\nSaved data/processed/hnsc_baseline.h5ad, results/tables/baseline_pca.csv and 4 figures")
