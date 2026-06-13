"""
Phase 1: Self-distillation training data for the visual DFlash drafter.

Samples token sequences directly from LlamaGen-3B at inference settings
(CFG 4.0, temperature 1.0, top-k 2000) so the drafter trains on exactly the
distribution it will draft at inference (DFlash paper recipe: train on
target-generated responses for alignment).

Output: sharded .npz files in --out, each with
  tokens: (n, 576) uint16   visual token sequences
  labels: (n,)    uint16    ImageNet class ids

Resumable: existing shard files are skipped.

Run:  python generate_training_data.py            (60 per class, ~16h)
      python generate_training_data.py --per-class 2 --batch 4 --out data/smoke
"""
import argparse
import gc
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "LlamaGen"))

from autoregressive.models.gpt import GPT_models  # noqa: E402


def cfg_combine(logits: torch.Tensor, cfg_scale: float) -> torch.Tensor:
    cond, uncond = torch.split(logits, logits.shape[0] // 2, dim=0)
    return uncond + (cond - uncond) * cfg_scale


def sample_topk(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    """logits (b, V) -> (b, 1) sampled token ids."""
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        thresh = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = torch.where(logits < thresh, torch.full_like(logits, -float("inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_batch(model, step_fn, class_ids: torch.Tensor, num_tokens: int,
                   cfg_scale: float, temperature: float, top_k: int) -> torch.Tensor:
    """class_ids (b,) -> (b, num_tokens) sampled visual token sequences."""
    b = class_ids.shape[0]
    device = class_ids.device
    cond_combined = torch.cat([class_ids, torch.full_like(class_ids, model.num_classes)])

    seq = torch.empty(b, num_tokens, dtype=torch.long, device=device)
    input_pos = torch.arange(0, 1, device=device)
    logits, _ = model(None, cond_combined, input_pos)
    next_token = sample_topk(cfg_combine(logits[:, -1], cfg_scale), temperature, top_k)
    seq[:, 0] = next_token[:, 0]

    input_pos = torch.tensor([1], device=device, dtype=torch.int)
    for i in range(1, num_tokens):
        x = next_token.view(b, 1)
        logits, _ = step_fn(torch.cat([x, x]), cond_idx=None, input_pos=input_pos)
        next_token = sample_topk(cfg_combine(logits[:, -1], cfg_scale), temperature, top_k)
        seq[:, i] = next_token[:, 0]
        input_pos += 1
    return seq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpt-model", default="GPT-3B")
    p.add_argument("--gpt-ckpt", default=os.path.join(ROOT, "LlamaGen", "pretrained_models", "c2i_3B_384.pt"))
    p.add_argument("--out", default=os.path.join(ROOT, "data", "train_tokens"))
    p.add_argument("--per-class", type=int, default=60)
    p.add_argument("--num-classes", type=int, default=1000,
                   help="sample only the first N classes (model always has 1000)")
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--shard-size", type=int, default=1200)
    p.add_argument("--codebook-size", type=int, default=16384)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--downsample-size", type=int, default=16)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--config", default=None, help="experiment json (overrides flags)")
    p.add_argument("--run-dir", default=None, help="if set, shards written to $run-dir/data")
    p.add_argument("--pretrained", default=None, help="cluster pretrained dir for *_ckpt_rel resolution")
    p.add_argument("--array-id", type=int, default=0)
    p.add_argument("--array-size", type=int, default=1)
    args = p.parse_args()

    if args.config:
        import json as _json
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = _json.load(f)
        assert cfg["task"] == "c2i", "use generate_training_data_t2i.py for t2i"
        tgt, dg, smp = cfg["target"], cfg["datagen"], cfg["sampling"]
        if args.pretrained:
            args.gpt_ckpt = os.path.join(args.pretrained, tgt["gpt_ckpt_rel"])
        args.gpt_model = tgt["gpt_model"]
        args.codebook_size = tgt["codebook_size"]
        args.image_size = tgt["image_size"]
        args.downsample_size = tgt["downsample_size"]
        args.cfg_scale = smp["cfg_scale"]
        args.temperature = smp["temperature"]
        args.top_k = smp["top_k"]
        args.per_class = dg["per_class"]
        args.num_classes = dg["num_classes"]
        args.batch = dg["batch"]
        args.shard_size = dg["shard_size"]
        args.seed = dg["seed"]
        args.no_compile = not dg.get("compile", True)
        if args.run_dir:
            args.out = os.path.join(args.run_dir, "data")
            os.makedirs(args.run_dir, exist_ok=True)
            with open(os.path.join(args.run_dir, "config.json"), "w", encoding="utf-8") as f:
                _json.dump(cfg, f, indent=2)

    assert torch.cuda.is_available()
    device = "cuda"
    os.makedirs(args.out, exist_ok=True)
    latent = args.image_size // args.downsample_size
    num_tokens = latent ** 2

    # deterministic shuffled job list -> stable shard boundaries for resume
    rng = np.random.default_rng(args.seed)
    jobs = np.repeat(np.arange(args.num_classes, dtype=np.uint16), args.per_class)
    rng.shuffle(jobs)
    num_shards = (len(jobs) + args.shard_size - 1) // args.shard_size
    all_pending = [s for s in range(num_shards)
                   if not os.path.exists(os.path.join(args.out, f"shard_{s:04d}.npz"))]
    # array-task partitioning: each task takes shards where s % array_size == array_id
    pending = [s for s in all_pending if s % args.array_size == args.array_id]
    print(f"{len(jobs)} sequences in {num_shards} shards; task {args.array_id}/{args.array_size} "
          f"-> {len(pending)} shards to generate", flush=True)
    if not pending:
        return

    print("Loading target model ...", flush=True)
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    model = GPT_models[args.gpt_model](vocab_size=args.codebook_size, block_size=num_tokens,
                                       num_classes=1000, cls_token_num=1,
                                       model_type="c2i")
    ckpt = torch.load(args.gpt_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt if "model" not in ckpt else ckpt["model"], strict=False)
    del ckpt
    gc.collect()
    model.to(device=device, dtype=torch.bfloat16).eval()
    torch.cuda.empty_cache()

    with torch.device(device):
        model.setup_caches(max_batch_size=2 * args.batch, max_seq_length=1 + num_tokens,
                           dtype=torch.bfloat16)
    print(f"GPU mem after setup: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    step_fn = model.forward
    if not args.no_compile:
        try:
            step_fn = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
            print("torch.compile enabled for decode step", flush=True)
        except Exception as e:
            print(f"torch.compile unavailable ({e}); using eager", flush=True)

    torch.manual_seed(args.seed)
    t_start = time.perf_counter()
    done_seqs = 0
    for shard_idx in pending:
        lo = shard_idx * args.shard_size
        hi = min(lo + args.shard_size, len(jobs))
        shard_jobs = jobs[lo:hi]
        toks = np.empty((len(shard_jobs), num_tokens), dtype=np.uint16)
        for off in range(0, len(shard_jobs), args.batch):
            chunk = shard_jobs[off: off + args.batch].astype(np.int64)
            n_real = len(chunk)
            if n_real < args.batch:  # pad to keep batch shape stable for caches/compile
                chunk = np.concatenate([chunk, np.zeros(args.batch - n_real, dtype=np.int64)])
            cls = torch.tensor(chunk, device=device)
            seq = generate_batch(model, step_fn, cls, num_tokens, args.cfg_scale,
                                 args.temperature, args.top_k)
            toks[off: off + n_real] = seq[:n_real].cpu().numpy().astype(np.uint16)
            done_seqs += n_real
        tmp = os.path.join(args.out, f"shard_{shard_idx:04d}.tmp.npz")
        np.savez_compressed(tmp, tokens=toks, labels=shard_jobs)
        os.replace(tmp, os.path.join(args.out, f"shard_{shard_idx:04d}.npz"))
        el = time.perf_counter() - t_start
        rate = done_seqs / el
        remain = (len(pending) * args.shard_size - done_seqs) / max(rate, 1e-9)
        print(f"shard {shard_idx:04d} done | {done_seqs} seqs | {rate:.2f} seq/s | "
              f"ETA {remain/3600:.1f} h", flush=True)
    print("DATAGEN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
