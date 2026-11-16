"""Step 03: clean GDC clinical data to one row per patient and attach it to the samples.

Input : metadata/clinical.tsv, metadata/exposure.tsv, metadata/follow_up.tsv (if present)
        data/interim/hnsc_counts.h5ad                    (from step 02)
Output: results/tables/clinical_clean.tsv                (one row per patient)
        data/processed/hnsc_annotated.h5ad               (counts + clinical columns in .obs)

Usage : python scripts/03_clean_clinical.py
"""
import os

import anndata as ad
import numpy as np
import pandas as pd

MISSING = ["'--", "Not Reported", "not reported", "Unknown", "unknown", "Not Applicable"]


def read(name):
    return pd.read_csv(f"metadata/{name}.tsv", sep="\t", dtype=str, na_values=MISSING)


def first_valid(s):
    """First non-missing value in a group (NaN if the whole group is missing)."""
    s = s.dropna()
    return s.iloc[0] if len(s) else np.nan


def simplify_stage(x):
    """'Stage IVA' -> 'IV', 'Stage II' -> 'II'."""
    if pd.isna(x):
        return np.nan
    x = x.replace("Stage", "").strip().upper()
    for s in ["IV", "III", "II", "I"]:  # longest first so 'IV' is not read as 'I'
        if x.startswith(s):
            return s
    return np.nan


def site_group(x):
    """Group detailed anatomical sites into the main head-and-neck regions."""
    if pd.isna(x):
        return np.nan
    x = x.lower()
    if "tonsil" in x or "oropharynx" in x or "base of tongue" in x:
        return "Oropharynx"
    if "hypopharynx" in x or "pyriform" in x:
        return "Hypopharynx"
    if "larynx" in x or "glottis" in x:
        return "Larynx"
    if any(k in x for k in ["tongue", "mouth", "gum", "cheek", "palate", "lip", "retromolar"]):
        return "Oral cavity"
    return "Other"


clin = read("clinical")
key = "cases.submitter_id"  # patient barcode, e.g. TCGA-CQ-7068
print(f"clinical.tsv: {len(clin)} rows for {clin[key].nunique()} patients")

# 1. Demographics: constant per patient, so take the first non-missing value
demo = clin.groupby(key).agg(
    age=("demographic.age_at_index", first_valid),
    sex=("demographic.sex_at_birth", first_valid),
    vital_status=("demographic.vital_status", first_valid),
    days_to_death=("demographic.days_to_death", first_valid),
)

# 2. Diagnosis: use only rows describing the primary disease
is_primary = clin["diagnoses.diagnosis_is_primary_disease"].str.lower() == "true"
diag = clin[is_primary].groupby(key).agg(
    primary_diagnosis=("diagnoses.primary_diagnosis", first_valid),
    site_detail=("diagnoses.tissue_or_organ_of_origin", first_valid),
    stage_pathologic=("diagnoses.ajcc_pathologic_stage", first_valid),
    stage_clinical=("diagnoses.ajcc_clinical_stage", first_valid),
    t_stage=("diagnoses.ajcc_pathologic_t", first_valid),
    n_stage=("diagnoses.ajcc_pathologic_n", first_valid),
    grade=("diagnoses.tumor_grade", first_valid),
)

# 3. Last follow-up: the latest day the patient was known to be alive
fu = pd.to_numeric(clin["diagnoses.days_to_last_follow_up"], errors="coerce").groupby(clin[key]).max()
if os.path.exists("metadata/follow_up.tsv"):
    f = read("follow_up")
    col = "follow_ups.days_to_follow_up"
    if col in f.columns:
        fu2 = pd.to_numeric(f[col], errors="coerce").groupby(f[key]).max()
        fu = pd.concat([fu, fu2], axis=1).max(axis=1)
fu = fu.rename("days_to_last_follow_up")

# 4. Treatment: did the patient receive radiation / drug therapy?
given = clin["treatments.treatment_or_therapy"].str.lower() == "yes"
ttype = clin["treatments.treatment_type"].fillna("")
treat = pd.DataFrame(
    {
        "radiation": (given & ttype.str.contains("Radiation")).groupby(clin[key]).any(),
        "drug_therapy": (given & ttype.str.contains("Pharmaceutical|Chemo")).groupby(clin[key]).any(),
    }
).astype(float)  # 1 = yes, 0 = no

# 5. Tobacco exposure
expo = read("exposure").groupby(key).agg(
    smoking_status=("exposures.tobacco_smoking_status", first_valid),
    pack_years=("exposures.pack_years_smoked", first_valid),
)

pat = demo.join([diag, fu, treat, expo], how="left")

# 6. Tidy types and derive analysis variables
for c in ["age", "days_to_death", "days_to_last_follow_up", "pack_years"]:
    pat[c] = pd.to_numeric(pat[c], errors="coerce")
pat["stage"] = pat["stage_pathologic"].map(simplify_stage)
pat["stage"] = pat["stage"].fillna(pat["stage_clinical"].map(simplify_stage))  # fall back to clinical stage
pat["site"] = pat["site_detail"].map(site_group)

# Overall survival: event = death; time = days to death, or to last follow-up if alive (censored)
pat["OS_event"] = (pat["vital_status"] == "Dead").astype(float)
pat.loc[pat["vital_status"].isna(), "OS_event"] = np.nan
pat["OS_time"] = np.where(pat["OS_event"] == 1, pat["days_to_death"], pat["days_to_last_follow_up"])
pat.loc[pat["OS_time"] <= 0, "OS_time"] = np.nan

os.makedirs("results/tables", exist_ok=True)
pat.to_csv("results/tables/clinical_clean.tsv", sep="\t")

print(f"\nPatients: {len(pat)}")
print("Missing values per column:\n", pat.isna().sum().to_string())
for c in ["sex", "vital_status", "stage", "site", "grade", "smoking_status"]:
    print(f"\n{c}:\n", pat[c].value_counts(dropna=False).to_string())
print("\nAge: median %.0f (range %.0f-%.0f)" % (pat["age"].median(), pat["age"].min(), pat["age"].max()))
ok = pat["OS_time"].notna() & pat["OS_event"].notna()
print("Usable for survival: %d patients, %d deaths, median follow-up %.0f days"
      % (ok.sum(), pat.loc[ok, "OS_event"].sum(), pat.loc[ok, "OS_time"].median()))

# 7. Attach clinical columns to every sample via the patient barcode
adata = ad.read_h5ad("data/interim/hnsc_counts.h5ad")
obs = adata.obs.join(pat, on="case_id")
print(f"\nSamples without clinical data: {int(obs['vital_status'].isna().sum())} of {len(obs)}")
for c in obs.columns:  # text columns -> categorical, so they save cleanly with missing values
    if obs[c].dtype == object:
        obs[c] = obs[c].astype("category")
adata.obs = obs
os.makedirs("data/processed", exist_ok=True)
adata.write_h5ad("data/processed/hnsc_annotated.h5ad")
print("Saved results/tables/clinical_clean.tsv and data/processed/hnsc_annotated.h5ad")
