"""Step 07: unsupervised clustering of tumours on the baseline and on Geneformer embeddings.

Input : results/tables/baseline_pca.csv                 (step 04)
        data/processed/emb_V1_mean.csv, emb_V2_cls.csv, emb_V2_mean.csv   (step 06; missing ones are skipped)
        metadata/gdc_sample_sheet*.tsv, results/tables/clinical_clean.tsv
Output: results/tables/sample_annotation.csv, k_selection.csv, umap_coords.csv,
        cluster_labels_k<K>.csv, cluster_metrics_k<K>.csv, ari_between_representations_k<K>.csv
        results/figures/07_*.png   (files that depend on k carry _k<K> in the name)

Usage : python scripts/07_cluster.py            # k chosen from the baseline's best silhouette
        python scripts/07_cluster.py --k 4      # force a number of clusters
"""
import argparse
import glob
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", module="umap")  # harmless notices about future UMAP defaults

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=None)
args = parser.parse_args()

FIG, TAB = "results/figures", "results/tables"
N_DIM, K_RANGE, SEED = 20, range(2, 9), 0
SOURCES = {
    "baseline": f"{TAB}/baseline_pca.csv",
    "GF_V1_mean": "data/processed/emb_V1_mean.csv",
    "GF_V2_cls": "data/processed/emb_V2_cls.csv",
    "GF_V2_mean": "data/processed/emb_V2_mean.csv",
}

# ---------- 1. Sample annotation ----------
sheet = pd.read_csv(glob.glob("metadata/gdc_sample_sheet*.tsv")[0], sep="\t")
ann = pd.DataFrame({
    "case_id": sheet["Case ID"].values,
    "tissue_type": sheet["Tissue Type"].values,
    "tumor_descriptor": sheet["Tumor Descriptor"].values,
}, index=pd.Index(sheet["Sample ID"].astype(str).values, name="sample_uid"))
ann = ann.join(pd.read_csv(f"{TAB}/clinical_clean.tsv", sep="\t", index_col=0), on="case_id")
ann.to_csv(f"{TAB}/sample_annotation.csv")

# ---------- 2. Load representations: every one becomes samples x 20 dimensions ----------
reps = {}
for name, path in SOURCES.items():
    if not os.path.exists(path):
        print(f"[skip] {name}: {path} not found")
        continue
    X = pd.read_csv(path, index_col=0)
    X.index = X.index.astype(str)
    X = X.loc[ann.index.intersection(X.index)]
    if name == "baseline":
        Z = X.values[:, :N_DIM]                     # already principal components
    else:                                           # standardise, then reduce to 20 components
        Z = PCA(n_components=N_DIM, random_state=SEED).fit_transform(StandardScaler().fit_transform(X.values))
    reps[name] = pd.DataFrame(Z, index=X.index)
    print(f"{name}: {X.shape[0]} samples, {X.shape[1]} original dimensions -> {N_DIM}")
names = list(reps)
common = ann.index
for n in names:
    common = common.intersection(reps[n].index)
ann = ann.loc[common]

# Cluster primary tumours only, one sample per patient
tum = ann[(ann["tissue_type"] == "Tumor") & (ann["tumor_descriptor"] == "Primary")]
tum = tum[~tum["case_id"].duplicated()]
print(f"\nAll samples: {len(ann)} | primary tumours used for clustering: {len(tum)}")

# ---------- 3. UMAP of all samples (sanity check: do normals separate?) ----------
coords = {}
for n in names:
    emb = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=SEED).fit_transform(reps[n].loc[ann.index].values)
    coords[n] = pd.DataFrame(emb, index=ann.index, columns=["UMAP1", "UMAP2"])
pd.concat(coords, axis=1).to_csv(f"{TAB}/umap_coords.csv")


def scatter(ax, xy, labels, title):
    cats = sorted(pd.Series(labels).dropna().astype(str).unique())
    lab = pd.Series(labels, index=xy.index).astype(object)
    for i, c in enumerate(cats):
        m = (lab.astype(str) == c).values
        ax.scatter(xy.values[m, 0], xy.values[m, 1], s=8, color=plt.cm.tab10(i % 10), label=c)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7, markerscale=1.5, loc="best", frameon=False)


fig, axes = plt.subplots(len(names), 2, figsize=(11, 4.2 * len(names)), squeeze=False)
for r, n in enumerate(names):
    scatter(axes[r, 0], coords[n], ann["tissue_type"], f"{n}: tissue type")
    scatter(axes[r, 1], coords[n], ann["site"], f"{n}: anatomical site")
plt.tight_layout()
plt.savefig(f"{FIG}/07_umap_all_samples.png", dpi=150)
plt.close()

# ---------- 4. Choose k: k-means for k = 2..8 ----------
ksel = []
for n in names:
    Z = reps[n].loc[tum.index].values
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        ksel.append({"representation": n, "k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(Z, km.labels_)})
ksel = pd.DataFrame(ksel)
ksel.to_csv(f"{TAB}/k_selection.csv", index=False)
print("\nSilhouette score by k (higher = better separated clusters):")
print(ksel.pivot(index="k", columns="representation", values="silhouette").round(3).to_string())

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for n in names:
    d = ksel[ksel["representation"] == n]
    ax[0].plot(d["k"], d["inertia"] / d["inertia"].iloc[0], "o-", label=n)
    ax[1].plot(d["k"], d["silhouette"], "o-", label=n)
ax[0].set(xlabel="k", ylabel="Inertia (relative to k=2)", title="Elbow plot")
ax[1].set(xlabel="k", ylabel="Mean silhouette", title="Silhouette score")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG}/07_k_selection.png", dpi=150)
plt.close()

base = ksel[ksel["representation"] == names[0]]
K = args.k or int(base.loc[base["silhouette"].idxmax(), "k"])
print(f"\nUsing k = {K}" + ("" if args.k else f" (best silhouette for {names[0]}; override with --k)"))

# ---------- 5. Three clustering methods per representation ----------
labels, metrics = pd.DataFrame(index=tum.index), []
site = tum["site"].astype(str).values
for n in names:
    Z = reps[n].loc[tum.index].values
    km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(Z)
    hc = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(Z)
    gm = GaussianMixture(n_components=K, covariance_type="diag", n_init=5, random_state=SEED).fit_predict(Z)
    labels[f"{n}_kmeans"], labels[f"{n}_hier"], labels[f"{n}_gmm"] = km, hc, gm
    # stability: does k-means find the same clusters from different random starts?
    runs = [KMeans(n_clusters=K, n_init=1, random_state=s).fit_predict(Z) for s in range(10)]
    stab = np.mean([adjusted_rand_score(runs[i], runs[j]) for i in range(10) for j in range(i + 1, 10)])
    metrics.append({
        "representation": n,
        "silhouette_kmeans": silhouette_score(Z, km),
        "silhouette_hier": silhouette_score(Z, hc),
        "silhouette_gmm": silhouette_score(Z, gm),
        "ARI_kmeans_vs_hier": adjusted_rand_score(km, hc),
        "ARI_kmeans_vs_gmm": adjusted_rand_score(km, gm),
        "kmeans_stability_ARI": stab,
        "ARI_kmeans_vs_site": adjusted_rand_score(km, site),
        "NMI_kmeans_vs_site": normalized_mutual_info_score(km, site),
        "cluster_sizes": " / ".join(map(str, np.bincount(km))),
    })
metrics = pd.DataFrame(metrics).set_index("representation")
labels.to_csv(f"{TAB}/cluster_labels_k{K}.csv")
metrics.to_csv(f"{TAB}/cluster_metrics_k{K}.csv")
print("\nClustering metrics:")
print(metrics.round(3).T.to_string())

# ---------- 6. Do the representations agree with each other? ----------
ari = pd.DataFrame([[adjusted_rand_score(labels[f"{a}_kmeans"], labels[f"{b}_kmeans"]) for b in names]
                    for a in names], index=names, columns=names)
ari.to_csv(f"{TAB}/ari_between_representations_k{K}.csv")
print("\nAgreement between representations (ARI of k-means labels):")
print(ari.round(3).to_string())

fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 1.4 * len(names) + 1.5))
im = ax.imshow(ari.values, vmin=0, vmax=1, cmap="viridis")
ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
ax.set_yticks(range(len(names)), names)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, f"{ari.values[i, j]:.2f}", ha="center", va="center",
                color="white" if ari.values[i, j] < 0.6 else "black")
plt.colorbar(im, label="Adjusted Rand index")
ax.set_title(f"Agreement of k-means clusters (k={K})")
plt.tight_layout()
plt.savefig(f"{FIG}/07_ari_between_representations_k{K}.png", dpi=150)
plt.close()

# ---------- 7. Figures: clusters on the UMAP, and dendrograms ----------
fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.2), squeeze=False)
for c, n in enumerate(names):
    scatter(axes[0, c], coords[n].loc[tum.index], labels[f"{n}_kmeans"].values, f"{n}: k-means clusters (k={K})")
plt.tight_layout()
plt.savefig(f"{FIG}/07_umap_clusters_k{K}.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 3.8), squeeze=False)
for c, n in enumerate(names):
    dendrogram(linkage(reps[n].loc[tum.index].values, method="ward"), ax=axes[0, c],
               truncate_mode="lastp", p=30, no_labels=True, color_threshold=0)
    axes[0, c].set_title(f"{n}: Ward dendrogram (top 30 branches)", fontsize=10)
    axes[0, c].set_ylabel("Merge distance")
plt.tight_layout()
plt.savefig(f"{FIG}/07_dendrograms.png", dpi=150)
plt.close()
print("\nSaved 6 tables in results/tables and 5 figures in results/figures (07_*.png)")
