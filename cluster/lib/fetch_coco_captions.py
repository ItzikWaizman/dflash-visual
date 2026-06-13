"""
Fetch COCO captions for t2i datagen + eval.

The cluster's HTTPS proxy whitelists huggingface.co but not cocodataset.org
(direct download fails with SSL hostname-mismatch), so we pull the captions
from a HuggingFace mirror of the Karpathy COCO split:

    yerevann/coco-karpathy   (parquet)

The Karpathy split is the standard image-captioning eval split used by t2i
speculative-decoding papers (LANTERN, LANTERN++, SJD++), so this is more
directly comparable than vanilla COCO 2017 annotations anyway.

Each parquet row has:
    cocoid:     int          # original COCO image_id
    split:      str          # "train" / "val" / "test" / "restval"
    sentences:  list[str]    # 5 captions per image

We flatten one row per caption and emit:
    $DFLASH_DATA/coco/captions_train2017.jsonl    (from Karpathy train, ~414K captions)
    $DFLASH_DATA/coco/captions_val2017.jsonl      (from Karpathy validation, ~25K captions)

Each line:  {"image_id": <int>, "caption": <str>}
"""
import argparse
import json
import os
import subprocess
import sys

from huggingface_hub import hf_hub_download


KARPATHY_REPO = "yerevann/coco-karpathy"
SPLIT_FILES = {
    "train": "data/train-00000-of-00001.parquet",
    "val":   "data/validation-00000-of-00001.parquet",
}


def _ensure_pyarrow():
    try:
        import pyarrow.parquet  # noqa: F401
        return
    except ImportError:
        pass
    print("[coco] pyarrow not present -- installing into current env", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "pyarrow>=15"]
    )
    import pyarrow.parquet  # noqa: F401


def parquet_to_jsonl(parquet_path: str, jsonl_path: str) -> int:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["cocoid", "sentences"])
    cocoids = table.column("cocoid").to_pylist()
    sentences = table.column("sentences").to_pylist()  # list[list[str]]

    n = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for cid, sents in zip(cocoids, sentences):
            for cap in sents:
                if not cap:
                    continue
                json.dump({"image_id": int(cid), "caption": cap.strip()}, f,
                          ensure_ascii=False)
                f.write("\n")
                n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True,
                   help="$DFLASH_DATA (e.g. /scratch300/$USER/dflash_vlm/data)")
    args = p.parse_args()

    coco_dir = os.path.join(args.out_dir, "coco")
    os.makedirs(coco_dir, exist_ok=True)

    targets = {
        "train": os.path.join(coco_dir, "captions_train2017.jsonl"),
        "val":   os.path.join(coco_dir, "captions_val2017.jsonl"),
    }

    if all(os.path.exists(p_) and os.path.getsize(p_) > 0 for p_ in targets.values()):
        print(f"[coco] both JSONLs already present:")
        for split, p_ in targets.items():
            print(f"  {split}: {p_} ({os.path.getsize(p_) / 1024:.0f} KB)")
        print("[coco] DONE")
        return

    _ensure_pyarrow()

    for split, jsonl_path in targets.items():
        if os.path.exists(jsonl_path) and os.path.getsize(jsonl_path) > 0:
            print(f"[coco] {jsonl_path} already present, skipping", flush=True)
            continue
        print(f"[coco] HF: {KARPATHY_REPO}/{SPLIT_FILES[split]}", flush=True)
        parquet_path = hf_hub_download(
            repo_id=KARPATHY_REPO,
            filename=SPLIT_FILES[split],
            repo_type="dataset",
        )
        n = parquet_to_jsonl(parquet_path, jsonl_path)
        sz_kb = os.path.getsize(jsonl_path) / 1024
        print(f"[coco] wrote {jsonl_path}  ({n} captions, {sz_kb:.0f} KB)",
              flush=True)

    print("[coco] DONE", flush=True)


if __name__ == "__main__":
    main()
