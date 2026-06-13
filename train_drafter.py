"""
Phase 2: Train the visual DFlash drafter on self-distilled LlamaGen-3B tokens.

Per training step (DFlash recipe):
  1. Frozen target forward (no-grad, bf16, full causal) over [cls, t_1..t_575]
     -> capture 5 hidden layers -> raw concat features for positions 0..575.
  2. Sample A random anchors p in [1, 561] per sequence. Each anchor defines a
     16-token block: anchor token t_p at position p + 15 mask embeddings at
     positions p+1..p+15.
  3. One drafter forward over all A blocks jointly: queries attend to fused ctx
     features at positions < p (per block) and bidirectionally within their own
     block (block-diagonal attention mask).
  4. Weighted CE on the 15 masked positions, w_j = exp(-(j-1)/gamma), gamma=7.

Shared frozen target tok_embeddings and LM head. Checkpoints + resume.
Periodic real-tau probe through the actual speculative engine.

Run:  python train_drafter.py --data data/train_tokens --epochs 4
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
    RawHiddenCapture, VisualDFlashDrafter, spec_generate_real,
)

BLOCK = 16
NUM_TOKENS = 576


def load_target(args, device):
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    model = GPT_models[args.gpt_model](vocab_size=args.codebook_size, block_size=NUM_TOKENS,
                                       num_classes=1000, cls_token_num=1, model_type="c2i")
    ckpt = torch.load(args.gpt_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt if "model" not in ckpt else ckpt["model"], strict=False)
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
    toks, labs = [], []
    for s in shards:
        z = np.load(s)
        toks.append(z["tokens"])
        labs.append(z["labels"])
    tokens = np.concatenate(toks)
    labels = np.concatenate(labs)
    print(f"dataset: {tokens.shape[0]} sequences from {len(shards)} shards", flush=True)
    return torch.from_numpy(tokens.astype(np.int64)), torch.from_numpy(labels.astype(np.int64))


def build_attn_mask(anchors: torch.Tensor, ctx_len: int, device) -> torch.Tensor:
    """anchors: (B, A) absolute positions p. Returns bool (B, 1, A*16, ctx_len + A*16)."""
    B, A = anchors.shape
    pos = torch.arange(ctx_len, device=device)
    ctx_vis = pos.view(1, 1, -1) < anchors.unsqueeze(-1)           # (B, A, ctx_len)
    ctx_vis = ctx_vis.unsqueeze(2).expand(B, A, BLOCK, ctx_len)    # (B, A, 16, ctx)
    ctx_vis = ctx_vis.reshape(B, A * BLOCK, ctx_len)
    self_vis = torch.block_diag(*([torch.ones(BLOCK, BLOCK, dtype=torch.bool, device=device)] * A))
    self_vis = self_vis.unsqueeze(0).expand(B, -1, -1)             # (B, A*16, A*16)
    return torch.cat([ctx_vis, self_vis], dim=-1).unsqueeze(1)


@torch.no_grad()
def extract_features(target, capture, tokens, labels, device):
    """tokens (B, 576) -> raw concat features (B, 576, 5D) for positions 0..575."""
    idx = tokens[:, : NUM_TOKENS - 1].to(device)          # t_1 .. t_575
    cond = labels.to(device)
    input_pos = torch.arange(0, NUM_TOKENS, device=device)
    target(idx, cond_idx=cond, input_pos=input_pos)
    return capture.concat(cond_only=False)                 # (B, 576, 5D)


def make_optimizer(drafter, lr, weight_decay):
    decay, no_decay = [], []
    for n, p in drafter.named_parameters():
        (no_decay if p.dim() < 2 else decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(groups, lr=lr, betas=(0.9, 0.95))
        print("optimizer: bitsandbytes AdamW8bit", flush=True)
    except Exception as e:
        opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))
        print(f"optimizer: torch AdamW (bf16 states) [{type(e).__name__}]", flush=True)
    return opt


def lr_lambda(step, total_steps, warmup_frac=0.04, min_ratio=0.1):
    warmup = max(1, int(total_steps * warmup_frac))
    if step < warmup:
        return step / warmup
    t = (step - warmup) / max(1, total_steps - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))


def run_probe(target, drafter, capture, device, classes=(207, 360, 88, 979)):
    """Real-tau probe via the actual speculative engine (greedy verification)."""
    drafter.eval()
    with torch.device(device):
        target.setup_caches(max_batch_size=2, max_seq_length=1 + NUM_TOKENS,
                            dtype=torch.bfloat16)
    taus = []
    for cls in classes:
        _, st = spec_generate_real(target, drafter, capture, cls, NUM_TOKENS,
                                   cfg_scale=4.0, block_size=BLOCK, mode="greedy",
                                   device=device)
        taus.append(st["tau"])
    # release caches so the training forward path (cache-free) works again
    for layer in target.layers:
        layer.attention.kv_cache = None
    if hasattr(target, "causal_mask"):
        del target.causal_mask
    torch.cuda.empty_cache()
    drafter.train()
    return sum(taus) / len(taus), taus


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpt-model", default="GPT-3B")
    p.add_argument("--gpt-ckpt", default=os.path.join(ROOT, "LlamaGen", "pretrained_models", "c2i_3B_384.pt"))
    p.add_argument("--codebook-size", type=int, default=16384)
    p.add_argument("--data", default=os.path.join(ROOT, "data", "train_tokens"))
    p.add_argument("--out", default=os.path.join(ROOT, "checkpoints"))
    p.add_argument("--num-layers", type=int, default=5)
    p.add_argument("--num-features", type=int, default=5)
    p.add_argument("--anchors", type=int, default=32)
    p.add_argument("--batch-seqs", type=int, default=2)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=7.0)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--probe-every", type=int, default=1500)
    p.add_argument("--max-steps", type=int, default=0, help="debug: stop after N optim steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default=None, help="experiment json (cluster mode)")
    p.add_argument("--run-dir", default=None, help="if set, data/, checkpoints/ under here")
    p.add_argument("--pretrained", default=None, help="cluster pretrained dir")
    args = p.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        assert cfg["task"] == "c2i", "use train_drafter_t2i.py for t2i targets"
        tgt, dr, tr = cfg["target"], cfg["drafter"], cfg["train"]
        if args.pretrained:
            args.gpt_ckpt = os.path.join(args.pretrained, tgt["gpt_ckpt_rel"])
        args.gpt_model = tgt["gpt_model"]
        args.codebook_size = tgt["codebook_size"]
        args.num_layers = dr["num_layers"]
        args.num_features = dr["num_features"]
        args.anchors = tr["anchors"]
        args.batch_seqs = tr["batch_seqs"]
        args.accum = tr["accum"]
        args.epochs = tr["epochs"]
        args.lr = tr["lr"]
        args.weight_decay = tr["weight_decay"]
        args.gamma = tr["gamma"]
        args.clip = tr["clip"]
        args.log_every = tr["log_every"]
        args.ckpt_every = tr["ckpt_every"]
        args.probe_every = tr["probe_every"]
        args.seed = tr["seed"]
        if args.run_dir:
            args.data = os.path.join(args.run_dir, "data")
            args.out = os.path.join(args.run_dir, "checkpoints")

    assert torch.cuda.is_available()
    device = "cuda"
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print("Loading target ...", flush=True)
    target = load_target(args, device)
    capture = RawHiddenCapture(target, args.num_features)
    print(f"feature layers: {capture.layer_ids} | GPU {torch.cuda.memory_allocated()/1e9:.1f} GB",
          flush=True)

    drafter = VisualDFlashDrafter(dim=target.config.dim, n_head=target.config.n_head,
                                  num_layers=args.num_layers, num_features=args.num_features,
                                  block_size=BLOCK).to(device=device, dtype=torch.bfloat16)
    drafter.train()
    n_params = sum(p_.numel() for p_ in drafter.parameters())
    print(f"drafter: {n_params/1e6:.0f}M trainable params", flush=True)

    tokens_all, labels_all = load_dataset(args.data)
    N = tokens_all.shape[0]
    steps_per_epoch = N // (args.batch_seqs * args.accum)
    total_steps = steps_per_epoch * args.epochs
    print(f"{steps_per_epoch} optim steps/epoch x {args.epochs} epochs = {total_steps}", flush=True)

    opt = make_optimizer(drafter, args.lr, args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total_steps))

    # resume
    start_step = 0
    latest = os.path.join(args.out, "latest.pt")
    if os.path.exists(latest):
        st = torch.load(latest, map_location=device, weights_only=False)
        drafter.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_step = st["step"]
        print(f"resumed from step {start_step}", flush=True)

    loss_w = torch.exp(-(torch.arange(BLOCK - 1, dtype=torch.float32)) / args.gamma).to(device)
    loss_w = loss_w / loss_w.mean()
    freqs = target.freqs_cis  # (577, hd//2, 2) on device
    ctx_freqs = freqs[:NUM_TOKENS]  # positions 0..575
    log_path = os.path.join(args.out, "train_log.jsonl")

    gen = torch.Generator()
    micro_per_step = args.batch_seqs * args.accum
    run_loss, run_acc, run_n = 0.0, torch.zeros(BLOCK - 1, device=device), 0
    t_last = time.perf_counter()
    seqs_done_window = 0

    optim_step = start_step
    while optim_step < total_steps:
        epoch = optim_step // steps_per_epoch
        gen.manual_seed(args.seed * 100003 + epoch)
        perm = torch.randperm(N, generator=gen)
        # skip already-consumed part of this epoch on resume
        in_epoch = optim_step - epoch * steps_per_epoch
        offset = in_epoch * micro_per_step

        while offset + micro_per_step <= N and optim_step < total_steps:
            opt.zero_grad(set_to_none=True)
            for micro in range(args.accum):
                sl = perm[offset: offset + args.batch_seqs]
                offset += args.batch_seqs
                toks = tokens_all[sl]
                labs = labels_all[sl]
                B = toks.shape[0]

                raw = extract_features(target, capture, toks, labs, device)
                fused_ctx = drafter.fuse(raw)  # grads flow into fc/hidden_norm

                anchors = torch.randint(1, NUM_TOKENS - BLOCK + 2, (B, args.anchors),
                                        device=device)  # p in [1, 561]
                toks_dev = toks.to(device)
                # block embeddings: anchor token + 15 masks
                anchor_tok = toks_dev.gather(1, anchors - 1)        # t_p  (B, A)
                anchor_emb = target.tok_embeddings(anchor_tok)      # (B, A, D)
                mask_emb = drafter.mask_embedding.to(anchor_emb.dtype)
                block_emb = torch.cat(
                    [anchor_emb.unsqueeze(2),
                     mask_emb.view(1, 1, 1, -1).expand(B, args.anchors, BLOCK - 1, -1)],
                    dim=2).reshape(B, args.anchors * BLOCK, -1)

                block_pos = anchors.unsqueeze(-1) + torch.arange(BLOCK, device=device)
                block_freqs = freqs[block_pos.reshape(B, -1)]       # (B, A*16, hd//2, 2)
                attn_mask = build_attn_mask(anchors, NUM_TOKENS, device)

                hid = drafter.forward_train(fused_ctx, ctx_freqs, block_emb,
                                            block_freqs, attn_mask)
                hid = hid.view(B, args.anchors, BLOCK, -1)[:, :, 1:, :]
                logits = target.output(drafter.norm(hid)).float()   # (B, A, 15, V)

                # labels: token at position p+j is t_{p+j} = tokens[:, p+j-1]
                lab_idx = (block_pos[:, :, 1:] - 1)                 # (B, A, 15)
                labels_blk = toks_dev.gather(
                    1, lab_idx.reshape(B, -1)).view(B, args.anchors, BLOCK - 1)

                ce = F.cross_entropy(logits.permute(0, 3, 1, 2), labels_blk,
                                     reduction="none")              # (B, A, 15)
                loss = (ce * loss_w.view(1, 1, -1)).mean()
                (loss / args.accum).backward()

                with torch.no_grad():
                    run_acc += (logits.argmax(-1) == labels_blk).float().mean(dim=(0, 1))
                    run_loss += loss.item()
                    run_n += 1
                seqs_done_window += B

            torch.nn.utils.clip_grad_norm_(drafter.parameters(), args.clip)
            opt.step()
            sched.step()
            optim_step += 1

            if optim_step % args.log_every == 0:
                acc = (run_acc / run_n).cpu().tolist()
                # i.i.d. tau proxy: 1 + sum_k prod_{j<=k} acc_j
                prod, tau_proxy = 1.0, 1.0
                for a in acc:
                    prod *= a
                    tau_proxy += prod
                dt = time.perf_counter() - t_last
                sps = seqs_done_window / dt
                rec = {"step": optim_step, "epoch": epoch,
                       "loss": run_loss / run_n, "acc1": acc[0], "acc8": acc[7],
                       "acc15": acc[14], "tau_proxy": tau_proxy,
                       "lr": sched.get_last_lr()[0], "seq_per_s": sps,
                       "eta_h": (total_steps - optim_step) * micro_per_step / sps / 3600}
                print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in rec.items()}), flush=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                run_loss, run_n = 0.0, 0
                run_acc.zero_()
                t_last = time.perf_counter()
                seqs_done_window = 0

            if optim_step % args.ckpt_every == 0 or optim_step == total_steps:
                tmp = latest + ".tmp"
                torch.save({"model": drafter.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "step": optim_step,
                            "args": vars(args)}, tmp)
                os.replace(tmp, latest)

            if optim_step % args.probe_every == 0:
                tau_mean, taus = run_probe(target, drafter, capture, device)
                rec = {"step": optim_step, "probe_tau": tau_mean,
                       "probe_taus": [round(t, 2) for t in taus]}
                print("PROBE " + json.dumps(rec), flush=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                t_last = time.perf_counter()
                seqs_done_window = 0

            if args.max_steps and optim_step - start_step >= args.max_steps:
                print("max-steps reached, stopping", flush=True)
                tmp = latest + ".tmp"
                torch.save({"model": drafter.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "step": optim_step,
                            "args": vars(args)}, tmp)
                os.replace(tmp, latest)
                return

    torch.save({"model": drafter.state_dict(), "step": optim_step, "args": vars(args)},
               os.path.join(args.out, "final.pt"))
    print("TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
