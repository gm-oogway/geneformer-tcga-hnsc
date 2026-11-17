"""Step 06: pass each tokenized sample through Geneformer and save one embedding vector per sample.

Input : data/tokenized/hnsc_<version>.dataset           (from step 05)
        Geneformer model folder (default: ~/tools/Geneformer/<model name>)
Output: data/processed/emb_<version>_mean.csv           (average of all gene-token embeddings)
        data/processed/emb_<version>_cls.csv            (V2 only: embedding at the <cls> token)

Usage : python scripts/06_extract_embeddings.py --version V1
        python scripts/06_extract_embeddings.py --version V2 --limit 10   # timing test
        python scripts/06_extract_embeddings.py --version V2              # full run (resumes if interrupted)
"""
import argparse
import os
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import BertForMaskedLM

MODELS = {"V1": "Geneformer-V1-10M", "V2": "Geneformer-V2-104M_CLcancer"}

parser = argparse.ArgumentParser()
parser.add_argument("--version", choices=["V1", "V2"], required=True)
parser.add_argument("--model-dir", default=None)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--layer", type=int, default=-2, help="-2 = second-to-last layer (Geneformer's default), -1 = last")
args = parser.parse_args()

model_dir = args.model_dir or str(Path.home() / "tools/Geneformer" / MODELS[args.version])
device = "cuda" if torch.cuda.is_available() else "cpu"
has_cls = args.version == "V2"  # V2 sequences are <cls> gene gene ... gene <eos>
kinds = ["mean", "cls"] if has_cls else ["mean"]
suffix = f"_test{args.limit}" if args.limit else ""
out = {k: f"data/processed/emb_{args.version}_{k}{suffix}.csv" for k in kinds}
os.makedirs("data/processed", exist_ok=True)

# 1. Data
ds = load_from_disk(f"data/tokenized/hnsc_{args.version}.dataset")
if args.limit:
    ds = ds.select(range(args.limit))
done = set()
if os.path.exists(out["mean"]) and not args.limit:  # resume an interrupted run
    done = set(pd.read_csv(out["mean"], usecols=[0]).iloc[:, 0].astype(str))
todo = [i for i in range(len(ds)) if ds[i]["sample_uid"] not in done]
print(f"Samples: {len(ds)} total, {len(done)} already done, {len(todo)} to run")

# 2. Model
model = BertForMaskedLM.from_pretrained(model_dir, output_hidden_states=True).to(device).eval()
cfg = model.config
print(f"Model: {model_dir}")
print(f"  layers={cfg.num_hidden_layers}, hidden size={cfg.hidden_size}, heads={cfg.num_attention_heads}, "
      f"parameters={sum(p.numel() for p in model.parameters()) / 1e6:.0f}M, device={device}")
pad_id = cfg.pad_token_id if cfg.pad_token_id is not None else 0


def flush(buffers):
    """Append finished samples to the CSV files, so an interrupted run loses nothing."""
    for k in kinds:
        if buffers[k]:
            uids, vecs = zip(*buffers[k])
            df = pd.DataFrame(np.vstack(vecs), index=pd.Index(uids, name="sample_uid"))
            df.columns = [f"e{j}" for j in range(df.shape[1])]
            df.to_csv(out[k], mode="a", header=not os.path.exists(out[k]))
            buffers[k].clear()


# 3. Forward passes
buffers = {k: [] for k in kinds}
start = time.time()
for b in range(0, len(todo), args.batch_size):
    rows = [ds[i] for i in todo[b: b + args.batch_size]]
    max_len = max(len(r["input_ids"]) for r in rows)
    ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    for j, r in enumerate(rows):
        n = len(r["input_ids"])
        ids[j, :n] = torch.tensor(r["input_ids"], dtype=torch.long)
        mask[j, :n] = 1
    with torch.inference_mode():  # no gradients: we only read the model, we do not train it
        hidden = model(input_ids=ids.to(device), attention_mask=mask.to(device)).hidden_states[args.layer]
    hidden = hidden.float().cpu()  # shape: (samples, tokens, hidden size)

    gene_mask = mask.clone()
    if has_cls:  # leave <cls> (first) and <eos> (last real token) out of the average
        gene_mask[:, 0] = 0
        for j, r in enumerate(rows):
            gene_mask[j, len(r["input_ids"]) - 1] = 0
    m = gene_mask.unsqueeze(-1).float()
    mean_emb = (hidden * m).sum(dim=1) / m.sum(dim=1)

    for j, r in enumerate(rows):
        buffers["mean"].append((r["sample_uid"], mean_emb[j].numpy()))
        if has_cls:
            buffers["cls"].append((r["sample_uid"], hidden[j, 0].numpy()))

    n_done = min(b + args.batch_size, len(todo))
    if n_done % 10 == 0 or n_done == len(todo):
        flush(buffers)
        per = (time.time() - start) / n_done
        peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print(f"  {n_done}/{len(todo)}  {per:.1f} s/sample  "
              f"remaining ~{per * (len(todo) - n_done) / 60:.0f} min  peak memory {peak_gb:.1f} GB", flush=True)
flush(buffers)

# 4. Report
for k in kinds:
    e = pd.read_csv(out[k], index_col=0)
    print(f"\n{out[k]}: {e.shape[0]} samples x {e.shape[1]} dimensions, any NaN: {bool(e.isna().any().any())}")
if not args.limit:
    print(f"Total time this run: {(time.time() - start) / 60:.1f} min")
else:
    per = (time.time() - start) / len(todo)
    print(f"\nEstimated time for all 566 samples: {per * 566 / 60:.0f} min ({per * 566 / 3600:.1f} h)")
