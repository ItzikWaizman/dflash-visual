"""
Phase 1 (t2i): Self-distillation training data for LlamaGen-T2I targets.

Differences vs c2i datagen:
  - Loads a FLAN-T5-XL encoder, computes left-padded text embeddings for each
    prompt (max 120 tokens).
  - Target model is LlamaGen-T2I-{XL|XXL}, conditioned on T5 features as a
    soft-prompt PREFIX (cls_token_num=120, not 1).
  - Stores tokens + prompt_id per sample; prompt strings + T5 features are
    cached to a separate npz so training never has to re-encode T5.
  - Resumable via shard-level idempotency. Job-array friendly: each task
    handles a disjoint shard range via SLURM_ARRAY_TASK_ID.

Output:
  $RUN/data/t5_features.npz         prompts (N,) str, feats (N,120,2048) bf16,
                                    masks (N,120) bool
  $RUN/data/shard_XXXX.npz          tokens (n, T) uint16, prompt_ids (n,) int32

Usage:
  python generate_training_data_t2i.py --config cluster/configs/<EXP>.json \
       --run-dir $DFLASH_RUNS/$EXP --pretrained $DFLASH_PRETRAINED \
       --array-id 0 --array-size 4
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "LlamaGen"))


def load_prompts(prompts_file: str, num_sequences: int, seed: int) -> list:
    """Read a jsonl with {'caption': str} per line; return the first
    `num_sequences` after a stable shuffle."""
    rng = np.random.default_rng(seed)
    captions = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cap = obj.get("caption") or obj.get("text") or obj.get("prompt")
            if cap:
                captions.append(cap.strip())
    if num_sequences > len(captions):
        # cycle with shuffle to reach target size
        reps = (num_sequences + len(captions) - 1) // len(captions)
        captions = captions * reps
    captions = np.array(captions, dtype=object)
    perm = rng.permutation(len(captions))
    return captions[perm][:num_sequences].tolist()


def cache_t5_features(prompts: list, t5_path: str, t5_model_type: str,
                       max_len: int, device: str, out_path: str):
    """Compute and save T5 features for all prompts. Run once per
    experiment; subsequent datagen tasks just mmap-load."""
    if os.path.exists(out_path):
        print(f"[t5] features already cached at {out_path}", flush=True)
        return
    from language.t5 import T5Embedder
    print(f"[t5] loading FLAN-T5-XL from {t5_path}", flush=True)
    t5 = T5Embedder(device=device, local_cache=True, cache_dir=t5_path,
                    dir_or_name=t5_model_type, torch_dtype=torch.bfloat16,
                    model_max_length=max_len)
    N = len(prompts)
    feat_dim = None
    feats = None
    masks = np.empty((N, max_len), dtype=bool)
    bs = 32
    for s in range(0, N, bs):
        chunk = prompts[s: s + bs]
        emb, m = t5.get_text_embeddings(chunk)  # emb (b,L,D), m (b,L) right-pad
        if feats is None:
            feat_dim = emb.shape[-1]
            feats = np.empty((N, max_len, feat_dim), dtype=np.float16)
        # left-padding (LlamaGen convention): keep mask flipped and roll emb so
        # valid tokens sit at the tail of the [0:max_len) slot
        m_flip = torch.flip(m, dims=[-1])
        rolled = torch.empty_like(emb)
        for i in range(emb.shape[0]):
            valid = int(m[i].sum().item())
            rolled[i] = torch.cat([emb[i, valid:], emb[i, :valid]], dim=0)
        feats[s: s + emb.shape[0]] = rolled.to(torch.float16).cpu().numpy()
        masks[s: s + emb.shape[0]] = m_flip.bool().cpu().numpy()
        if (s // bs) % 32 == 0:
            print(f"[t5] encoded {s + emb.shape[0]} / {N}", flush=True)
    print(f"[t5] saving to {out_path}", flush=True)
    np.savez_compressed(out_path, prompts=np.array(prompts, dtype=object),
                        feats=feats.astype(np.float16), masks=masks)
    del t5
    gc.collect()
    torch.cuda.empty_cache()


def cfg_combine(logits: torch.Tensor, cfg_scale: float) -> torch.Tensor:
    cond, uncond = torch.split(logits, logits.shape[0] // 2, dim=0)
    return uncond + (cond - uncond) * cfg_scale


def sample_topk(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        thresh = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = torch.where(logits < thresh, torch.full_like(logits, -float("inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_batch_t2i(model, step_fn, c_embed: torch.Tensor, c_mask: torch.Tensor,
                       num_tokens: int, cls_token_num: int, cfg_scale: float,
                       temperature: float, top_k: int) -> torch.Tensor:
    """c_embed (b, 120, 2048) bf16; returns (b, num_tokens) sampled image tokens."""
    b = c_embed.shape[0]
    device = c_embed.device
    # CFG: cond half = real text, uncond half = handled internally by CaptionEmbedder
    # via force_drop_ids=1 -> learned unconditional embedding.
    cond_combined = torch.cat([c_embed, c_embed], dim=0)
    seq = torch.empty(b, num_tokens, dtype=torch.long, device=device)

    input_pos = torch.arange(0, cls_token_num, device=device)
    # encode null prompt for the uncond half via the model's drop path
    uncond_mask = torch.cat([torch.zeros(b, dtype=torch.long, device=device),
                             torch.ones(b, dtype=torch.long, device=device)])
    cond_emb_dropped = model.cls_embedding(cond_combined, train=False,
                                           force_drop_ids=uncond_mask)
    h = model.tok_dropout(cond_emb_dropped[:, :cls_token_num])
    # manual prefill: replicate the cond_idx-only branch of Transformer.forward
    freqs = model.freqs_cis.to(device)
    bs2 = h.shape[0]
    mask = model.causal_mask[:bs2, None, input_pos]
    freqs_cis = freqs[input_pos]
    for layer in model.layers:
        h = layer(h, freqs_cis, input_pos, mask)
    logits = model.output(model.norm(h)).float()
    next_token = sample_topk(cfg_combine(logits[:, -1], cfg_scale), temperature, top_k)
    seq[:, 0] = next_token[:, 0]

    input_pos = torch.tensor([cls_token_num], device=device, dtype=torch.int)
    for i in range(1, num_tokens):
        x = next_token.view(b, 1)
        logits, _ = step_fn(torch.cat([x, x]), cond_idx=None, input_pos=input_pos)
        next_token = sample_topk(cfg_combine(logits[:, -1], cfg_scale), temperature, top_k)
        seq[:, i] = next_token[:, 0]
        input_pos += 1
    return seq


def load_target(target_cfg, pretrained_dir, num_tokens, device):
    from autoregressive.models.gpt import GPT_models
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    model = GPT_models[target_cfg["gpt_model"]](
        block_size=num_tokens,
        cls_token_num=target_cfg["cls_token_num"],
        model_type="t2i",
    )
    ckpt_path = os.path.join(pretrained_dir, target_cfg["gpt_ckpt_rel"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model") or ckpt.get("module") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(sd, strict=False)
    del ckpt
    gc.collect()
    model.to(device=device, dtype=torch.bfloat16).eval()
    torch.cuda.empty_cache()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to experiment json")
    p.add_argument("--run-dir", required=True, help="$DFLASH_RUNS/$EXP")
    p.add_argument("--pretrained", required=True, help="$DFLASH_PRETRAINED")
    p.add_argument("--data-root", default=None,
                   help="parent of prompts_file; default = $DFLASH_DATA")
    p.add_argument("--array-id", type=int, default=0)
    p.add_argument("--array-size", type=int, default=1)
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["task"] == "t2i", "use generate_training_data.py for c2i"

    data_root = args.data_root or os.environ.get("DFLASH_DATA",
                                                 os.path.join(args.pretrained, ".."))
    out_dir = os.path.join(args.run_dir, "data")
    os.makedirs(out_dir, exist_ok=True)

    # snapshot the config alongside the run
    with open(os.path.join(args.run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    device = "cuda"
    assert torch.cuda.is_available()
    tgt = cfg["target"]
    enc = cfg["text_encoder"]
    dg = cfg["datagen"]
    smp = cfg["sampling"]
    latent = tgt["image_size"] // tgt["downsample_size"]
    num_tokens = latent ** 2
    cls_tok = tgt["cls_token_num"]

    # ---- prompts + T5 feature cache (only rank 0 builds, others wait) ----
    prompts_file = os.path.join(data_root, cfg["datagen"]["prompts_file_rel"])
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"prompts file missing: {prompts_file}")
    prompts = load_prompts(prompts_file, dg["num_sequences"], dg["seed"])
    t5_cache = os.path.join(out_dir, "t5_features.npz")
    if args.array_id == 0:
        cache_t5_features(prompts, os.path.join(args.pretrained, enc["cache_rel"]),
                          enc["model"], enc["feature_max_len"], device, t5_cache)
    else:
        while not os.path.exists(t5_cache):
            print(f"[task {args.array_id}] waiting for t5_features.npz ...", flush=True)
            time.sleep(30)
        time.sleep(5)  # ensure write is complete

    z = np.load(t5_cache, allow_pickle=True)
    # Direct fp16 -> bf16 conversion: avoids the 60 GB fp32 detour that would
    # otherwise dominate CPU RAM for 60K-prompt runs (~30 GB feats + ~30 GB bf16).
    feats_all = torch.from_numpy(z["feats"]).to(torch.bfloat16)
    masks_all = torch.from_numpy(z["masks"])
    print(f"[task {args.array_id}] loaded T5 features {tuple(feats_all.shape)}",
          flush=True)

    # ---- target model ----
    print(f"[task {args.array_id}] loading target {tgt['gpt_model']}", flush=True)
    model = load_target(tgt, args.pretrained, num_tokens, device)
    with torch.device(device):
        model.setup_caches(max_batch_size=2 * dg["batch"],
                           max_seq_length=cls_tok + num_tokens,
                           dtype=torch.bfloat16)
    print(f"[task {args.array_id}] GPU mem after setup: "
          f"{torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    step_fn = model.forward
    if dg.get("compile", True):
        try:
            step_fn = torch.compile(model.forward, mode="reduce-overhead",
                                    fullgraph=False)
            print(f"[task {args.array_id}] torch.compile enabled", flush=True)
        except Exception as e:
            print(f"[task {args.array_id}] compile unavailable: {e}", flush=True)

    # ---- shard plan (idempotent across tasks) ----
    N = dg["num_sequences"]
    num_shards = (N + dg["shard_size"] - 1) // dg["shard_size"]
    my_shards = [s for s in range(num_shards) if s % args.array_size == args.array_id]
    my_shards = [s for s in my_shards
                 if not os.path.exists(os.path.join(out_dir, f"shard_{s:04d}.npz"))]
    print(f"[task {args.array_id}] {len(my_shards)} shards to do "
          f"out of {num_shards} total", flush=True)
    if not my_shards:
        return

    torch.manual_seed(dg["seed"] * 1000003 + args.array_id)
    t0 = time.perf_counter()
    done_seqs = 0
    for shard_idx in my_shards:
        lo = shard_idx * dg["shard_size"]
        hi = min(lo + dg["shard_size"], N)
        prompt_ids = np.arange(lo, hi, dtype=np.int32)
        toks = np.empty((len(prompt_ids), num_tokens), dtype=np.uint16)
        for off in range(0, len(prompt_ids), dg["batch"]):
            idxs = prompt_ids[off: off + dg["batch"]]
            n_real = len(idxs)
            if n_real < dg["batch"]:
                idxs = np.concatenate([idxs, np.zeros(dg["batch"] - n_real,
                                                     dtype=np.int32)])
            c_emb = feats_all[idxs].to(device)
            seq = generate_batch_t2i(model, step_fn, c_emb,
                                     masks_all[idxs].to(device),
                                     num_tokens, cls_tok, smp["cfg_scale"],
                                     smp["temperature"], smp["top_k"])
            toks[off: off + n_real] = seq[:n_real].cpu().numpy().astype(np.uint16)
            done_seqs += n_real
        tmp = os.path.join(out_dir, f"shard_{shard_idx:04d}.tmp.npz")
        np.savez_compressed(tmp, tokens=toks, prompt_ids=np.arange(lo, hi, dtype=np.int32))
        os.replace(tmp, os.path.join(out_dir, f"shard_{shard_idx:04d}.npz"))
        el = time.perf_counter() - t0
        rate = done_seqs / max(el, 1e-9)
        remain = (sum(min(dg["shard_size"], N - s * dg["shard_size"])
                      for s in my_shards) - done_seqs) / max(rate, 1e-9)
        print(f"[task {args.array_id}] shard {shard_idx:04d} done | "
              f"{done_seqs} seqs | {rate:.2f} seq/s | ETA {remain/3600:.1f} h",
              flush=True)
    print(f"[task {args.array_id}] DATAGEN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
