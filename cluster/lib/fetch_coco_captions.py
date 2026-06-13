"""
Download COCO 2017 captions and emit JSONL files (one caption per line).

Outputs:
    $DFLASH_DATA/coco/captions_train2017.jsonl   (~7M lines... actually ~591K)
    $DFLASH_DATA/coco/captions_val2017.jsonl     (~25K)

We download the official ~250 MB annotations zip via `requests` (HTTPS), which
on this cluster goes through the same proxy as pip / huggingface_hub. We never
write the train2017 images themselves.

Idempotent: skips both downloads and JSONL conversion if files already exist.
"""
import argparse
import io
import json
import os
import sys
import zipfile

import requests


COCO_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True,
                   help="$DFLASH_DATA (e.g. /scratch300/$USER/dflash_vlm/data)")
    args = p.parse_args()

    coco_dir = os.path.join(args.out_dir, "coco")
    os.makedirs(coco_dir, exist_ok=True)

    train_jsonl = os.path.join(coco_dir, "captions_train2017.jsonl")
    val_jsonl = os.path.join(coco_dir, "captions_val2017.jsonl")
    if os.path.exists(train_jsonl) and os.path.exists(val_jsonl):
        sz_t = os.path.getsize(train_jsonl) / 1024
        sz_v = os.path.getsize(val_jsonl) / 1024
        print(f"[coco] both JSONLs present already: "
              f"train={sz_t:.0f} KB  val={sz_v:.0f} KB", flush=True)
        print("[coco] DONE")
        return

    print(f"[coco] GET {COCO_URL}", flush=True)
    with requests.get(COCO_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        print(f"[coco]   Content-Length: {total / (1<<20):.1f} MB", flush=True)
        buf = io.BytesIO()
        got = 0
        next_mark = 32 << 20
        for chunk in r.iter_content(chunk_size=1 << 20):
            if not chunk:
                continue
            buf.write(chunk)
            got += len(chunk)
            if got >= next_mark:
                print(f"[coco]   ...{got / (1<<20):.0f} MB", flush=True)
                next_mark += 32 << 20
        print(f"[coco]   downloaded {got / (1<<20):.1f} MB", flush=True)

    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        for split in ("train", "val"):
            jsonl_path = os.path.join(coco_dir, f"captions_{split}2017.jsonl")
            if os.path.exists(jsonl_path) and os.path.getsize(jsonl_path) > 0:
                print(f"[coco] {jsonl_path} already present, skipping",
                      flush=True)
                continue
            entry = f"annotations/captions_{split}2017.json"
            print(f"[coco] extracting {entry}", flush=True)
            with zf.open(entry) as f:
                data = json.load(f)
            n = 0
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for ann in data["annotations"]:
                    json.dump({"image_id": ann["image_id"],
                               "caption": ann["caption"]}, f)
                    f.write("\n")
                    n += 1
            print(f"[coco] wrote {jsonl_path}  ({n} captions)", flush=True)

    print("[coco] DONE", flush=True)


if __name__ == "__main__":
    main()
