#!/usr/bin/env bash
# Run the complete pipeline, steps 01-08, from the repository root.
#
# Usage:  bash scripts/run_all.sh                  # everything (V2 embeddings take ~1.5 h on CPU)
#         SKIP_DOWNLOAD=1 bash scripts/run_all.sh  # data/raw already present
#
# Requires the conda environments `gf` (analysis) and `gdc` (download); see README.
set -euo pipefail
cd "$(dirname "$0")/.."
eval "$(conda shell.bash hook)"
conda activate gf

step() { echo; echo "=== $1  ($(date '+%H:%M:%S')) ==="; }

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  step "01 download"; bash scripts/01_download.sh
fi
step "02 build matrix";     python scripts/02_build_matrix.py
step "03 clinical data";    python scripts/03_clean_clinical.py
step "04 QC + baseline";    python scripts/04_qc_baseline.py
step "05 tokenize V1";      python scripts/05_tokenize.py --version V1
step "05 tokenize V2";      python scripts/05_tokenize.py --version V2
step "06 embeddings V1";    python scripts/06_extract_embeddings.py --version V1
step "06 embeddings V2";    python scripts/06_extract_embeddings.py --version V2
for k in 2 4; do
  step "07 clustering k=$k";        python scripts/07_cluster.py --k "$k"
  step "08 characterisation k=$k";  python scripts/08_characterise_clusters.py --k "$k"
done
step "done"
