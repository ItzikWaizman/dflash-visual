"""
Phase 2 (t2i): Train the visual DFlash drafter for LlamaGen-T2I targets.

Differences vs c2i training:
  - Target's `cls_token_num` = 120 (text prefix), so target hidden states have
    120 text positions followed by the image positions. The drafter captures
    only the IMAGE-position hidden states (positions 120..120+N-1) as its
    context features.
  - Conditioning is fed via `cond_idx = T5 features` (B, 120, 2048), not class
    ids. T5 features are precomputed and cached at $RUN/data/t5_features.npz.
  - Block positions are in image-coords (0..N-1); RoPE lookups offset by 120.

Usage:
  python train_drafter_t2i.py --config cluster/configs/<EXP>.json \
       --run-dir $DFLASH_RUNS/$EXP --pretrained $DFLASH_PRETRAINED
"""
import argparse
import gc
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "LlamaGen"))

from autoregressive.models.gpt import GPT_models  # noqa: E402
from dflash_visual_drafter import (  # noqa: E402
    RawHiddenCapture, VisualDFlashDrafter,
)


def load_target(gpt_model, gpt_ckpt, num_tokens, cls_token_num, device):
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    model = GPT_models[gpt_model](block_size=num_tokens, cls_token_num=cls_token_num,
                                  model_type="t2i")
    ckpt = torch.load(gpt_ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("model") or ckpt.get("module") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(sd, strict=False)
    del ckpt
    gc.collect()
    model.to(device=device, dtype=torch.bfloat16).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.freqs_cis = model.freqs_cis.to(device)
    torch.cuda.empty_cache()
    return model


def load_dataset(data_dir):
    shards = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
    assert shards, f"no shards found in {data_dir}"
    toks, pids = [], []
    for s in shards:
        z = np.load(s)
        toks.append(z["tokens"])
        pids.append(z["prompt_ids"])
    tokens = np.concatenate(toks)
    prompt_ids = np.concatenate(pids)
    print(f"dataset: {tokens.shape[0]} sequences from {len(shards)} shards", flush=True)
    return torch.from_numpy(tokens.astype(np.int64)), torch.from_numpy(prompt_ids.astype(np.int64))


def build_attn_mask(anchors_img, ctx_len, block, device):
    """anchors_img (B, A): IMAGE-coord positions. ctx is image positions 0..ctx_len-1."""
    B, A = anchors_img.shape
    pos = torch.arange(ctx_len, device=device)
    ctx_vis = pos.view(1, 1, -1) < anchors_img.unsqueeze(-1)             # (B,A,ctx)
    ctx_vis = ctx_vis.unsqueeze(2).expand(B, A, block, ctx_len).reshape(B, A * block, ctx_len)
    self_vis = torch.block_diag(*([torch.ones(block, block, dtype=torch.bool, device=device)] * A))
    self_vis = self_vis.unsqueeze(0).expand(B, -1, -1)
    return torch.cat([ctx_vis, self_vis], dim=-1).unsqueeze(1)


@torch.no_grad()
def extract_features_t2i(target, capture, tokens, t5_feats, num_tokens, cls_token_num, device,
                          return_target_argmax=False):
    """tokens (B, N); t5_feats (B, 120, 2048) bf16. Returns IMAGE-position hidden
    states (B, N, 5D) — text positions stripped.

    If return_target_argmax=True, also returns the target's argmax over its
    full-sequence logits (B, cls_token_num + N - 1) so the train loop can log
    `acc_vs_target_argmax` — the metric that actually predicts real speculative-
    decoding tau (see Debug Mode session a22afb, hypothesis H_M).
    """
    idx = tokens[:, : num_tokens - 1].to(device)
    cond = t5_feats.to(device)
    seq_len = cls_token_num + idx.shape[1]
    input_pos = torch.arange(seq_len, device=device)
    logits_all, _ = target(idx, cond_idx=cond, input_pos=input_pos, targets=None)
    raw_all = capture.concat(cond_only=False)
    feats = raw_all[:, cls_token_num:, :]
    if return_target_argmax:
        # argmax is tiny (B, seq_len) int64 — keep around; drop fp32 logits ASAP.
        target_argmax_all = logits_all.argmax(-1)
        return feats, target_argmax_all
    return feats


def make_optimizer(drafter, lr, wd):
    """torch.AdamW with fp32 master state. We deliberately do NOT use
    bnb.AdamW8bit anymore: when the model was stored in bf16, Adam updates of
    magnitude ~lr=4e-4 around 1.0 (RMSNorm gains) fell below bf16 mantissa
    precision (~2^-7=0.0078) and were silently rounded to zero. Confirmed by
    Debug Mode session a22afb (H_J): all 22 RMSNorm.weight vectors stayed at
    init=1.0 with std=0 after 15K steps. Drafter is now in fp32 master copies;
    keep the optimizer state in fp32 too."""
    decay, no_decay = [], []
    for _, p in drafter.named_parameters():
        (no_decay if p.dim() < 2 else decay).append(p)
    groups = [{"params": decay, "weight_decay": wd},
              {"params": no_decay, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))
    print("optimizer: torch AdamW (fp32 master)", flush=True)
    return opt


def lr_lambda(step, total, warmup_frac=0.04, min_ratio=0.1):
    warmup = max(1, int(total * warmup_frac))
    if step < warmup:
        return step / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--pretrained", required=True)
    p.add_argument("--max-steps", type=int, default=0)
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["task"] == "t2i"

    tgt, dr, tr = cfg["target"], cfg["drafter"], cfg["train"]
    latent = tgt["image_size"] // tgt["downsample_size"]
    NUM_TOKENS = latent ** 2
    BLOCK = dr["block_size"]
    cls_tok = tgt["cls_token_num"]

    device = "cuda"
    torch.manual_seed(tr["seed"])

    print("[t2i train] loading target", flush=True)
    target = load_target(tgt["gpt_model"],
                         os.path.join(args.pretrained, tgt["gpt_ckpt_rel"]),
                         NUM_TOKENS, cls_tok, device)
    capture = RawHiddenCapture(target, dr["num_features"])
    print(f"[t2i train] feature layers {capture.layer_ids} | "
          f"GPU {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    # Keep drafter parameters in FP32 (master copies) — Debug Mode session
    # a22afb confirmed bf16 storage swallowed RMSNorm gain updates (H_J).
    # autocast(bf16) below makes the forward run at bf16 speed without sacrificing
    # the precision of the underlying parameter updates.
    drafter = VisualDFlashDrafter(dim=target.config.dim, n_head=target.config.n_head,
                                  num_layers=dr["num_layers"],
                                  num_features=dr["num_features"],
                                  block_size=BLOCK).to(device=device)
    drafter.train()
    n_params = sum(p_.numel() for p_ in drafter.parameters())
    print(f"[t2i train] drafter: {n_params/1e6:.0f}M trainable params", flush=True)

    # ---- dataset + T5 cache ----
    data_dir = os.path.join(args.run_dir, "data")
    tokens_all, prompt_ids_all = load_dataset(data_dir)
    t5_cache = os.path.join(data_dir, "t5_features.npz")
    z = np.load(t5_cache, allow_pickle=True)
    # Direct fp16 -> bf16 load: avoids the ~60 GB fp32 detour on 60K-prompt runs.
    t5_feats_all = torch.from_numpy(z["feats"]).to(torch.bfloat16)
    print(f"[t2i train] T5 feats {tuple(t5_feats_all.shape)}", flush=True)

    N = tokens_all.shape[0]
    micro = tr["batch_seqs"] * tr["accum"]
    steps_per_epoch = N // micro
    total_steps = steps_per_epoch * tr["epochs"]
    print(f"[t2i train] {steps_per_epoch} steps/epoch x {tr['epochs']} epochs = {total_steps}",
          flush=True)

    opt = make_optimizer(drafter, tr["lr"], tr["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total_steps))

    out_dir = os.path.join(args.run_dir, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    latest = os.path.join(out_dir, "latest.pt")
    start = 0
    if os.path.exists(latest):
        st = torch.load(latest, map_location=device, weights_only=False)
        drafter.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start = st["step"]
        print(f"[t2i train] resumed step {start}", flush=True)

    loss_w = torch.exp(-(torch.arange(BLOCK - 1, dtype=torch.float32)) / tr["gamma"]).to(device)
    loss_w = loss_w / loss_w.mean()

    freqs = target.freqs_cis                               # (cls + N, hd//2, 2)
    ctx_freqs = freqs[cls_tok: cls_tok + NUM_TOKENS - 1]   # IMAGE positions 0..N-2

    log_path = os.path.join(args.run_dir, "logs", "train_log.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    gen = torch.Generator()
    run_loss = 0.0
    run_acc_data = torch.zeros(BLOCK - 1, device=device)
    run_acc_target = torch.zeros(BLOCK - 1, device=device)
    run_n = 0
    t_last, seen = time.perf_counter(), 0

    optim_step = start
    while optim_step < total_steps:
        epoch = optim_step // steps_per_epoch
        gen.manual_seed(tr["seed"] * 100003 + epoch)
        perm = torch.randperm(N, generator=gen)
        offset = (optim_step - epoch * steps_per_epoch) * micro

        while offset + micro <= N and optim_step < total_steps:
            opt.zero_grad(set_to_none=True)
            for _ in range(tr["accum"]):
                sl = perm[offset: offset + tr["batch_seqs"]]
                offset += tr["batch_seqs"]
                toks = tokens_all[sl]                  # (B, N) image tokens
                pids = prompt_ids_all[sl].numpy()
                t5_feats = t5_feats_all[pids]          # (B, 120, 2048)
                B = toks.shape[0]

                # Capture target features AND target's full-sequence argmax so we
                # can log acc_vs_target during training (Debug Mode H_M: this is
                # the metric that actually predicts speculative-decoding tau).
                raw, target_argmax_all = extract_features_t2i(
                    target, capture, toks, t5_feats, NUM_TOKENS, cls_tok, device,
                    return_target_argmax=True,
                )

                # autocast runs the drafter's compute in bf16 (fast on A100/H100)
                # while parameters and gradients remain fp32 (fixes H_J).
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    fused_ctx = drafter.fuse(raw)

                    ctx_len = NUM_TOKENS - 1
                    # anchors in image coords; anchor p means block [p+1..p+BLOCK],
                    # context = image positions 0..p
                    anchors = torch.randint(0, ctx_len - BLOCK + 1, (B, tr["anchors"]),
                                            device=device)
                    toks_dev = toks.to(device)
                    anchor_tok = toks_dev.gather(1, anchors)
                    anchor_emb = target.tok_embeddings(anchor_tok)
                    mask_emb = drafter.mask_embedding.to(anchor_emb.dtype)
                    block_emb = torch.cat([
                        anchor_emb.unsqueeze(2),
                        mask_emb.view(1, 1, 1, -1).expand(B, tr["anchors"], BLOCK - 1, -1)],
                        dim=2).reshape(B, tr["anchors"] * BLOCK, -1)

                    block_pos_img = anchors.unsqueeze(-1) + torch.arange(BLOCK, device=device)
                    block_freqs = freqs[(cls_tok + block_pos_img).reshape(B, -1)]
                    attn_mask = build_attn_mask(anchors, ctx_len, BLOCK, device)

                    hid = drafter.forward_train(fused_ctx, ctx_freqs, block_emb,
                                                block_freqs, attn_mask)
                    # forward_train already applies self.norm(x) at its end; the
                    # previous code re-applied drafter.norm here which (a) doubled
                    # the norm vs the inference path's single norm and (b) would
                    # cause a train/eval mismatch once norm.weight starts moving
                    # (fp32 fix above). Match inference: drop the extra norm.
                    hid = hid.view(B, tr["anchors"], BLOCK, -1)[:, :, 1:, :]
                    logits = target.output(hid).float()                  # (B,A,15,V)

                    lab_idx = block_pos_img[:, :, 1:]                    # (B,A,15)
                    labels_blk = toks_dev.gather(1, lab_idx.reshape(B, -1)
                                                 ).view(B, tr["anchors"], BLOCK - 1)

                    ce = F.cross_entropy(logits.permute(0, 3, 1, 2), labels_blk, reduction="none")
                    loss = (ce * loss_w.view(1, 1, -1)).mean()

                (loss / tr["accum"]).backward()

                with torch.no_grad():
                    # acc vs the sampled training token (old metric; H_M shows it
                    # ceilings around 0.25 due to broad target distribution)
                    drafter_pred = logits.argmax(-1)
                    run_acc_data += (drafter_pred == labels_blk).float().mean(dim=(0, 1))
                    # acc vs target's own argmax at the same positions — the real
                    # spec-decoding metric (H_M)
                    seq_pos_target = cls_tok + block_pos_img[:, :, :-1]    # (B, A, 15)
                    target_argmax_at_lab = torch.gather(
                        target_argmax_all, dim=1, index=seq_pos_target.reshape(B, -1),
                    ).view(B, tr["anchors"], BLOCK - 1)
                    run_acc_target += (drafter_pred == target_argmax_at_lab).float().mean(dim=(0, 1))
                    run_loss += loss.item()
                    run_n += 1
                seen += B

            torch.nn.utils.clip_grad_norm_(drafter.parameters(), tr["clip"])
            opt.step()
            sched.step()
            optim_step += 1

            if optim_step % tr["log_every"] == 0:
                acc_data = (run_acc_data / run_n).cpu().tolist()
                acc_target = (run_acc_target / run_n).cpu().tolist()
                # tau proxy from the RIGHT metric (vs target argmax)
                prod, tau_target = 1.0, 1.0
                for a in acc_target:
                    prod *= a
                    tau_target += prod
                # tau proxy from old metric (vs data sample) — kept for comparison
                prod, tau_data = 1.0, 1.0
                for a in acc_data:
                    prod *= a
                    tau_data += prod
                # RMSNorm gain health check (H_J): track if gains actually move
                norm_w_std = float(drafter.norm.weight.detach().float().std().item())
                norm_w_max = float(drafter.norm.weight.detach().float().max().item())
                norm_w_min = float(drafter.norm.weight.detach().float().min().item())
                dt = time.perf_counter() - t_last
                sps = seen / dt
                rec = {
                    "step": optim_step, "epoch": epoch,
                    "loss": run_loss / run_n,
                    "acc1_target": acc_target[0],  "acc8_target": acc_target[7],  "acc15_target": acc_target[14],
                    "acc1_data":   acc_data[0],    "acc8_data":   acc_data[7],    "acc15_data":   acc_data[14],
                    "tau_target": tau_target, "tau_data": tau_data,
                    "norm_w_std": norm_w_std, "norm_w_min": norm_w_min, "norm_w_max": norm_w_max,
                    "lr": sched.get_last_lr()[0], "seq_per_s": sps,
                    "eta_h": (total_steps - optim_step) * micro / sps / 3600,
                }
                print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in rec.items()}), flush=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                run_loss, run_n = 0.0, 0
                run_acc_data.zero_()
                run_acc_target.zero_()
                t_last, seen = time.perf_counter(), 0

            if optim_step % tr["ckpt_every"] == 0 or optim_step == total_steps:
                tmp = latest + ".tmp"
                torch.save({"model": drafter.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "step": optim_step,
                            "config": cfg}, tmp)
                os.replace(tmp, latest)

            if args.max_steps and optim_step - start >= args.max_steps:
                print("[t2i train] max-steps reached", flush=True)
                tmp = latest + ".tmp"
                torch.save({"model": drafter.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "step": optim_step,
                            "config": cfg}, tmp)
                os.replace(tmp, latest)
                return

    torch.save({"model": drafter.state_dict(), "step": optim_step, "config": cfg},
               os.path.join(out_dir, "final.pt"))
    print("[t2i train] TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
