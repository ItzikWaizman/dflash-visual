"""
Download LlamaGen + FLAN-T5-XL pretrained weights into $DFLASH_PRETRAINED.

Uses huggingface_hub (same HTTPS stack as pip / transformers) so it picks up
any cluster-level proxy or HF token config that already works.

Idempotent: re-running just verifies presence and downloads anything missing.

Usage (from any node with internet -- typically the login node):
    python cluster/lib/download_weights.py --out-dir $DFLASH_PRETRAINED
"""
import argparse
import os
import sys

from huggingface_hub import hf_hub_download


# LlamaGen weights are split across two HF repos: c2i checkpoints live under
# the FoundationVision org, t2i checkpoints live under the original
# peizesun/llamagen_t2i mirror (per LlamaGen's README, model zoo section).
LLAMAGEN_REPOS = {
    "FoundationVision/LlamaGen": [
        "c2i_3B_384.pt",        # ~11.8 GB -- c2i pilot target
        "c2i_XXL_384.pt",       # ~5.4 GB  -- alt c2i scale point
        "vq_ds16_c2i.pt",       # ~0.3 GB  -- VQ tokenizer for c2i
    ],
    "peizesun/llamagen_t2i": [
        "t2i_XL_stage2_512.pt", # LlamaGen-XL Stage II (512x512, 32x32 grid)
        "vq_ds16_t2i.pt",       # VQ tokenizer for t2i
    ],
}

# FLAN-T5-XL: keep the pytorch_model.bin shards (T5Embedder hardcodes those).
T5_FILES = [
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer_config.json",
    "tokenizer.json",
    "pytorch_model.bin.index.json",
    "pytorch_model-00001-of-00002.bin",
    "pytorch_model-00002-of-00002.bin",
]


def fetch(repo_id: str, filename: str, local_dir: str) -> str:
    target = os.path.join(local_dir, filename)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        sz_mb = os.path.getsize(target) / (1024 * 1024)
        print(f"  [skip] {filename}  ({sz_mb:.1f} MB already present)", flush=True)
        return target
    print(f"  [get]  {filename}  from {repo_id}", flush=True)
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    sz_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  [ok]   {filename}  ({sz_mb:.1f} MB)", flush=True)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True,
                   help="$DFLASH_PRETRAINED (e.g. /scratch300/$USER/dflash_vlm/pretrained)")
    p.add_argument("--skip-llamagen", action="store_true")
    p.add_argument("--skip-t5", action="store_true")
    p.add_argument("--t5-repo", default="google/flan-t5-xl")
    args = p.parse_args()

    if not args.skip_llamagen:
        llamagen_dir = os.path.join(args.out_dir, "llamagen")
        os.makedirs(llamagen_dir, exist_ok=True)
        print(f"\n=== LlamaGen weights -> {llamagen_dir}")
        for repo_id, files in LLAMAGEN_REPOS.items():
            for f in files:
                try:
                    fetch(repo_id, f, llamagen_dir)
                except Exception as e:
                    print(f"  [FAIL] {f}: {type(e).__name__}: {e}", flush=True)
                    sys.exit(1)

    if not args.skip_t5:
        t5_dir = os.path.join(args.out_dir, "t5", "flan-t5-xl")
        os.makedirs(t5_dir, exist_ok=True)
        print(f"\n=== FLAN-T5-XL -> {t5_dir}")
        for f in T5_FILES:
            try:
                fetch(args.t5_repo, f, t5_dir)
            except Exception as e:
                print(f"  [FAIL] {f}: {type(e).__name__}: {e}", flush=True)
                sys.exit(1)

    print("\n[downloads] DONE", flush=True)


if __name__ == "__main__":
    main()
