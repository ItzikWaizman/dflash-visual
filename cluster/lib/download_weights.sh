#!/bin/bash
# Download LlamaGen pretrained weights from HuggingFace into
# $DFLASH_PRETRAINED. Run this on the LOGIN node, not via sbatch -- on most
# clusters compute nodes have no outbound internet but login nodes do.
#
# Usage (from the repo root, on the login node):
#   bash cluster/lib/download_weights.sh
#
# Optional: pass model names to skip the others, e.g.:
#   bash cluster/lib/download_weights.sh c2i_3B_384 t2i_XL_stage2_512
set -euo pipefail

source /scratch300/$USER/env.sh
module load anaconda
conda activate /scratch300/$USER/conda_envs/unlearning

test -f cluster/env.sh || { echo "ERROR: run from /scratch300/\$USER/dflash_vlm/dflash-visual" >&2; exit 1; }
source ./cluster/env.sh

cd "$DFLASH_PRETRAINED/llamagen" 2>/dev/null || { mkdir -p "$DFLASH_PRETRAINED/llamagen" && cd "$DFLASH_PRETRAINED/llamagen"; }

ALL=(c2i_3B_384 c2i_XXL_384 vq_ds16_c2i t2i_XL_stage2_512 vq_ds16_t2i)
if [ $# -gt 0 ]; then
    WANT=("$@")
else
    WANT=("${ALL[@]}")
fi

download() {
    local name="$1"
    local file="${name}.pt"
    if [ -s "$file" ]; then
        echo "  [skip] $file already present ($(du -h "$file" | cut -f1))"
        return 0
    fi
    local url="https://huggingface.co/peizesun/llamagen/resolve/main/$file"
    echo "  [get]  $file"
    # -c resume, --tries=3, fail fast on 4xx so we don't write empty files
    if wget -c --tries=3 --retry-on-http-error=503 -O "$file" "$url"; then
        echo "  [ok]   $file  $(du -h "$file" | cut -f1)"
    else
        rm -f "$file"
        echo "  [FAIL] $file -- delete and retry" >&2
        return 1
    fi
}

for w in "${WANT[@]}"; do
    download "$w"
done

echo
echo "[downloads] DONE. Contents of $DFLASH_PRETRAINED/llamagen:"
ls -lh "$DFLASH_PRETRAINED/llamagen"
