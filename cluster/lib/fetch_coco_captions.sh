#!/bin/bash
# Download COCO 2017 captions and convert to JSONL (one caption per line).
# Run once before t2i datagen. ~250 MB download, ~700K train / 25K val captions.
#
# Usage:
#   sbatch -A gpu-tad-wolf_v2 -p gpu-tad-pool --qos=owner \
#          --time=1:00:00 --cpus-per-task=2 --mem=8G \
#          --chdir /scratch300/$USER/dflash_visual/code \
#          -o /scratch300/$USER/dflash_visual/fetch_coco.log \
#          cluster/scripts/fetch_coco_captions.sh
set -euo pipefail

source /scratch300/$USER/env.sh
module load anaconda
conda activate /scratch300/$USER/conda_envs/unlearning

test -f cluster/env.sh || { echo "ERROR: cluster/env.sh not found. Did sbatch get --chdir to the repo root?" >&2; exit 1; }
source ./cluster/env.sh

OUT="$DFLASH_DATA/coco"
mkdir -p "$OUT"
cd "$OUT"

if [ ! -f annotations_trainval2017.zip ] && [ ! -d annotations ]; then
    echo "[coco] downloading captions"
    wget -q --show-progress http://images.cocodataset.org/annotations/annotations_trainval2017.zip
fi

if [ ! -f annotations/captions_train2017.json ]; then
    unzip -q annotations_trainval2017.zip "annotations/captions_train2017.json" \
                                          "annotations/captions_val2017.json"
fi

python - <<'PY'
import json, os
root = os.environ["DFLASH_DATA"] + "/coco"
for split in ["train", "val"]:
    src = f"{root}/annotations/captions_{split}2017.json"
    dst = f"{root}/captions_{split}2017.jsonl"
    if os.path.exists(dst):
        print(f"[coco] {dst} already exists ({os.path.getsize(dst)//1024} KB)")
        continue
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(dst, "w", encoding="utf-8") as f:
        for ann in data["annotations"]:
            json.dump({"image_id": ann["image_id"], "caption": ann["caption"]}, f)
            f.write("\n")
    print(f"[coco] wrote {dst}  ({len(data['annotations'])} captions)")
PY

rm -f annotations_trainval2017.zip
echo "[coco] DONE"
