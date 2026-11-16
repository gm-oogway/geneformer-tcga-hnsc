"""Step 05: convert each sample into Geneformer's input format (rank value encoding).

Input : data/processed/hnsc_annotated.h5ad         (raw counts, from step 03)
Output: data/tokenized/hnsc_<version>.dataset      (one list of gene tokens per sample)

Usage : python scripts/05_tokenize.py --version V1 --limit 10   # dry run
        python scripts/05_tokenize.py --version V1              # all samples, small model
        python scripts/05_tokenize.py --version V2              # all samples, V2 models
"""
import argparse
import inspect
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import scanpy as sc
from datasets import load_from_disk

import geneformer
from geneformer import TranscriptomeTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--version", choices=["V1", "V2"], required=True)
parser.add_argument("--limit", type=int, default=None)
args = parser.parse_args()

# Each model family has its own dictionaries and input length
SETTINGS = {
    "V1": dict(sub="gene_dictionaries_30m", tag="gc30M", input_size=2048, special_token=False),
    "V2": dict(sub="", tag="gc104M", input_size=4096, special_token=True),
}[args.version]


def dict_dir():
    """Find the folder holding Geneformer's dictionary files."""
    for base in [Path(geneformer.__file__).parent, Path.home() / "tools/Geneformer/geneformer"]:
        d = base / SETTINGS["sub"]
        if (d / f"token_dictionary_{SETTINGS['tag']}.pkl").exists():
            return d
    raise FileNotFoundError("Geneformer dictionary files not found")


D = dict_dir()
token_file = D / f"token_dictionary_{SETTINGS['tag']}.pkl"
median_file = D / f"gene_median_dictionary_{SETTINGS['tag']}.pkl"
mapping_file = D / f"ensembl_mapping_dict_{SETTINGS['tag']}.pkl"
print(f"Dictionaries: {D}")

# 1. Prepare the tokenizer's input: a folder containing one .h5ad with raw counts
adata = sc.read_h5ad("data/processed/hnsc_annotated.h5ad")
if args.limit:
    adata = adata[: args.limit].copy()
adata.X = adata.X.astype(np.float32)
adata.obs["sample_uid"] = adata.obs_names.astype(str)  # unique label carried through tokenizing
adata.obs = adata.obs[["sample_uid", "n_counts"]]       # keep only what the tokenizer needs
in_dir = Path(f"data/tokenizer_input_{args.version}")
shutil.rmtree(in_dir, ignore_errors=True)
in_dir.mkdir(parents=True)
adata.write_h5ad(in_dir / "hnsc.h5ad")

# 2. Tokenize
kwargs = dict(
    custom_attr_name_dict={"sample_uid": "sample_uid"},
    nproc=1,
    model_input_size=SETTINGS["input_size"],
    special_token=SETTINGS["special_token"],
    gene_median_file=median_file,
    token_dictionary_file=token_file,
    gene_mapping_file=mapping_file,
)
if "model_version" in inspect.signature(TranscriptomeTokenizer.__init__).parameters:
    kwargs["model_version"] = args.version  # newer Geneformer releases take this argument
tk = TranscriptomeTokenizer(**kwargs)

suffix = f"_test{args.limit}" if args.limit else ""
prefix = f"hnsc_{args.version}{suffix}"
os.makedirs("data/tokenized", exist_ok=True)
tk.tokenize_data(str(in_dir), "data/tokenized", prefix, file_format="h5ad")

# 3. Inspect the result
ds = load_from_disk(f"data/tokenized/{prefix}.dataset")
print("\n", ds)
lengths = [len(x) for x in ds["input_ids"]]
print(f"Tokens per sample: min {min(lengths)}, max {max(lengths)}")

with open(token_file, "rb") as f:
    token_dict = pickle.load(f)
id_of_token = {v: k for k, v in token_dict.items()}
name_of_id = dict(zip(adata.var["ensembl_id"], adata.var["gene_name"]))
in_vocab = sum(g in token_dict for g in adata.var["ensembl_id"])
print(f"Genes in our data that are in the model vocabulary: {in_vocab} of {adata.n_vars}")

for i in range(min(2, len(ds))):
    ids = [id_of_token[t] for t in ds[i]["input_ids"][:16]]
    print(f"\nSample {ds[i]['sample_uid']} - first 16 tokens (highest ranked):")
    print("  ", [name_of_id.get(g, g) for g in ids])
print(f"\nSaved data/tokenized/{prefix}.dataset")
