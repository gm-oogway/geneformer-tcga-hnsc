#!/usr/bin/env bash
# Step 01: download TCGA-HNSC STAR-Counts files listed in the GDC manifest.
# Usage: bash scripts/01_download.sh
set -euo pipefail
MANIFEST=$(ls metadata/gdc_manifest*.txt | head -1)
mkdir -p data/raw
conda run -n gdc --no-capture-output gdc-client download -m "$MANIFEST" -d data/raw/
echo "Files downloaded: $(find data/raw -name '*.tsv' | wc -l)"
