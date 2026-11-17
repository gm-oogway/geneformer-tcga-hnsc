"""Step 08: what do the clusters mean? Clinical associations, survival, technical checks and marker genes.

Input : results/tables/cluster_labels_k<K>.csv        (step 07)
        results/tables/sample_annotation.csv           (step 07)
        data/processed/hnsc_baseline.h5ad              (step 04: log-CPM expression + QC metrics)
Output: results/tables/08_k<K>_association_tests.csv   (cluster vs clinical/technical variables)
        results/tables/08_k<K>_survival.csv            (log-rank and Cox results)
        results/tables/08_k<K>_markers_<rep>.csv       (top marker genes per cluster)
        results/tables/08_k<K>_cluster_profiles.csv    (readable summary of every cluster)
        results/figures/08_k<K>_*.png

Usage : python scripts/08_characterise_clusters.py --k 2
        python scripts/08_characterise_clusters.py --k 4
"""
import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=FutureWarning)
parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, required=True)
args = parser.parse_args()
K, FIG, TAB = args.k, "results/figures", "results/tables"

# Well-known gene programmes in head and neck cancer, used to give clusters a biological reading
PROGRAMMES = {
    "HPV proxy (CDKN2A / p16)": ["CDKN2A"],
    "Immune / T cell": ["CD8A", "GZMB", "PRF1", "CXCL9", "CXCL10"],
    "Stroma / mesenchymal": ["VIM", "COL1A1", "COL1A2", "FN1", "SPARC"],
    "Squamous differentiation": ["KRT5", "KRT14", "KRT6A", "DSG3", "TP63"],
    "Proliferation": ["MKI67", "TOP2A", "CCNB1", "MCM2"],
    "Hypoxia": ["CA9", "VEGFA", "PGK1", "SLC2A1"],
}

# ---------- 1. Load ----------
labels = pd.read_csv(f"{TAB}/cluster_labels_k{K}.csv", index_col=0)
labels.index = labels.index.astype(str)
ann = pd.read_csv(f"{TAB}/sample_annotation.csv", index_col=0).loc[labels.index]
adata = sc.read_h5ad("data/processed/hnsc_baseline.h5ad")
adata.obs_names = adata.obs_names.astype(str)
adata = adata[labels.index].copy()
adata.var_names = adata.var["gene_name"].astype(str).values
adata.var_names_make_unique()
adata.uns["log1p"] = {"base": None}  # X holds log(1 + CPM) from step 04

ann["library_size_M"] = adata.obs["library_size_M"].values
ann["n_genes_detected"] = adata.obs["n_genes_detected"].values
ann["smoking"] = ann["smoking_status"].map(
    lambda s: np.nan if pd.isna(s) else "Never" if "Non-Smoker" in s else "Current" if s == "Current Smoker" else "Former")
ann["stage_num"] = ann["stage"].map({"I": 1, "II": 2, "III": 3, "IV": 4})
ann["OS_years"] = ann["OS_time"] / 365.25
ann["collection_site"] = ann.index.str[5:7]  # TCGA tissue source site (hospital): a classic batch variable

# Programme scores: average z-score of the programme's genes (log-CPM)
X = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X),
                 index=adata.obs_names, columns=adata.var_names)
Zs = (X - X.mean()) / X.std().replace(0, np.nan)
for prog, genes in PROGRAMMES.items():
    present = [g for g in genes if g in Zs.columns]
    ann[prog] = Zs[present].mean(axis=1) if present else np.nan

reps = sorted({c.rsplit("_", 1)[0] for c in labels.columns if c.endswith("_kmeans")},
              key=lambda r: (r != "baseline", r))
CATEGORICAL = ["site", "stage", "sex", "smoking", "grade", "collection_site"]
CONTINUOUS = ["age", "library_size_M", "n_genes_detected"] + list(PROGRAMMES)
print(f"k = {K} | {len(ann)} primary tumours | representations: {', '.join(reps)}")

# ---------- 2. Association tests ----------
tests = []
for rep in reps:
    cl = labels[f"{rep}_kmeans"].astype(str)
    for var in CATEGORICAL:
        d = pd.DataFrame({"cl": cl, "v": ann[var]}).dropna()
        d = d[d["v"].map(d["v"].value_counts()) >= 5]  # drop categories with fewer than 5 patients
        tab = pd.crosstab(d["cl"], d["v"])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        chi2, p, _, _ = stats.chi2_contingency(tab)
        cramers_v = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
        tests.append({"representation": rep, "variable": var, "test": "chi-square",
                      "effect": "Cramer's V", "effect_size": cramers_v, "p": p})
    for var in CONTINUOUS:
        groups = [ann.loc[cl == c, var].dropna() for c in sorted(cl.unique())]
        if sum(len(g) > 0 for g in groups) < 2:
            continue
        H, p = stats.kruskal(*groups)
        n = sum(len(g) for g in groups)
        eps2 = (H - len(groups) + 1) / (n - len(groups))  # epsilon-squared effect size
        tests.append({"representation": rep, "variable": var, "test": "Kruskal-Wallis",
                      "effect": "epsilon^2", "effect_size": eps2, "p": p})
tests = pd.DataFrame(tests)
tests["p_adj_BH"] = multipletests(tests["p"], method="fdr_bh")[1]  # correct for running many tests
tests["significant"] = tests["p_adj_BH"] < 0.05
tests.to_csv(f"{TAB}/08_k{K}_association_tests.csv", index=False)
pd.set_option("display.width", 200)
print("\nSignificant associations (BH-adjusted p < 0.05), strongest first:")
sig = tests[tests["significant"]].sort_values(["representation", "effect_size"], ascending=[True, False])
print(sig[["representation", "variable", "effect", "effect_size", "p_adj_BH"]].round(4).to_string(index=False)
      if len(sig) else "  none")

# ---------- 3. Survival ----------
surv_rows = []
fig, axes = plt.subplots(1, len(reps), figsize=(4.8 * len(reps), 4), squeeze=False)
for i, rep in enumerate(reps):
    d = ann[["OS_years", "OS_event", "age", "stage_num"]].assign(cl=labels[f"{rep}_kmeans"].astype(str))
    d = d.dropna(subset=["OS_years", "OS_event"])
    d = d[d["OS_years"] > 0]
    ax = axes[0, i]
    for c in sorted(d["cl"].unique()):
        m = d["cl"] == c
        KaplanMeierFitter().fit(d.loc[m, "OS_years"], d.loc[m, "OS_event"],
                                label=f"cluster {c} (n={m.sum()})").plot_survival_function(ax=ax, ci_show=False, show_censors=True,
                                                                           censor_styles={"ms": 4, "marker": "|"})
    lr = multivariate_logrank_test(d["OS_years"], d["cl"], d["OS_event"])
    ax.set(title=f"{rep}  (log-rank p = {lr.p_value:.3g})", xlabel="Years (display cut at 10; few patients beyond)",
           ylabel="Overall survival", ylim=(0, 1.02), xlim=(0, 10))
    ax.title.set_fontsize(10)

    # Cox model: cluster effect adjusted for age and stage; reference = largest cluster
    ref = d["cl"].value_counts().idxmax()
    cox_d = d.dropna(subset=["age", "stage_num"])
    dummies = pd.get_dummies(cox_d["cl"], prefix="cluster", dtype=float).drop(columns=f"cluster_{ref}")
    cox_in = pd.concat([cox_d[["OS_years", "OS_event", "age", "stage_num"]], dummies], axis=1)
    cph = CoxPHFitter().fit(cox_in, duration_col="OS_years", event_col="OS_event")
    for term, r in cph.summary.iterrows():
        surv_rows.append({"representation": rep, "logrank_p": lr.p_value, "n": len(cox_in),
                          "deaths": int(cox_in["OS_event"].sum()), "term": term, "reference_cluster": ref,
                          "hazard_ratio": r["exp(coef)"], "HR_lower95": r["exp(coef) lower 95%"],
                          "HR_upper95": r["exp(coef) upper 95%"], "cox_p": r["p"]})
plt.tight_layout()
plt.savefig(f"{FIG}/08_k{K}_kaplan_meier.png", dpi=150)
plt.close()
surv = pd.DataFrame(surv_rows)
surv.to_csv(f"{TAB}/08_k{K}_survival.csv", index=False)
print("\nSurvival (Cox model adjusted for age and stage; hazard ratio > 1 = worse survival than reference):")
print(surv[["representation", "logrank_p", "term", "hazard_ratio", "HR_lower95", "HR_upper95", "cox_p"]]
      .round(3).to_string(index=False))

# ---------- 4. Marker genes per cluster ----------
for rep in reps:
    a = adata.copy()
    a.obs["cluster"] = labels[f"{rep}_kmeans"].astype(str).astype("category").values
    sc.tl.rank_genes_groups(a, "cluster", method="wilcoxon", use_raw=False)
    mk = sc.get.rank_genes_groups_df(a, group=None)
    mk = mk[(mk["pvals_adj"] < 0.05) & (mk["logfoldchanges"] > 0)]
    top = mk.groupby("group").head(25)
    top.to_csv(f"{TAB}/08_k{K}_markers_{rep}.csv", index=False)
    print(f"\n{rep}: top 12 marker genes per cluster (higher in that cluster than in the rest)")
    for g, d in top.groupby("group"):
        print(f"  cluster {g}: {', '.join(d['names'].head(12))}")

# ---------- 5. Programme scores by cluster ----------
progs = list(PROGRAMMES)
fig, axes = plt.subplots(len(reps), len(progs), figsize=(2.6 * len(progs), 2.6 * len(reps)), squeeze=False)
for r, rep in enumerate(reps):
    cl = labels[f"{rep}_kmeans"].astype(str)
    cats = sorted(cl.unique())
    for c, prog in enumerate(progs):
        ax = axes[r, c]
        ax.boxplot([ann.loc[cl == k, prog].dropna() for k in cats], showfliers=False)
        ax.set_xticks(range(1, len(cats) + 1), cats)
        if r == 0:
            ax.set_title(prog, fontsize=9)
        if c == 0:
            ax.set_ylabel(f"{rep}\nscore (z)", fontsize=9)
        if r == len(reps) - 1:
            ax.set_xlabel("cluster", fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG}/08_k{K}_programme_scores.png", dpi=150)
plt.close()

# HPV proxy by anatomical site (independent of clustering: tests the oropharynx observation directly)
fig, ax = plt.subplots(figsize=(6, 4))
sites = [s for s in ["Oral cavity", "Larynx", "Hypopharynx", "Oropharynx"] if (ann["site"] == s).sum() > 0]
ax.boxplot([X.loc[ann["site"] == s, "CDKN2A"] for s in sites], showfliers=True)
ax.set_xticks(range(1, len(sites) + 1), [f"{s}\n(n={(ann['site'] == s).sum()})" for s in sites])
ax.set(ylabel="CDKN2A expression, log(1 + CPM)", title="CDKN2A (p16, HPV surrogate) by tumour site")
plt.tight_layout()
plt.savefig(f"{FIG}/08_cdkn2a_by_site.png", dpi=150)
plt.close()

# ---------- 6. One readable profile per cluster ----------
rows = []
for rep in reps:
    cl = labels[f"{rep}_kmeans"].astype(str)
    for c in sorted(cl.unique()):
        m = cl == c
        d = ann[m]
        row = {"representation": rep, "cluster": c, "n": int(m.sum()),
               "median_age": d["age"].median(),
               "% male": 100 * (d["sex"] == "male").mean(),
               "% stage IV": 100 * (d["stage"] == "IV").mean(),
               "% current smoker": 100 * (d["smoking"] == "Current").mean(),
               "% died": 100 * d["OS_event"].mean(),
               "median library size (M)": d["library_size_M"].median()}
        for s in ["Oral cavity", "Larynx", "Oropharynx"]:
            row[f"% {s}"] = 100 * (d["site"] == s).mean()
        for p in progs:
            row[p] = d[p].mean()
        rows.append(row)
prof = pd.DataFrame(rows)
prof.to_csv(f"{TAB}/08_k{K}_cluster_profiles.csv", index=False)
print("\nCluster profiles (programme columns are mean z-scores: >0 above the cohort average):")
print(prof.round(2).T.to_string(header=False))
print(f"\nSaved tables results/tables/08_k{K}_*.csv and figures results/figures/08_*.png")
