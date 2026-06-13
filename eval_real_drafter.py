"""
Phase 3: Evaluate the trained visual DFlash drafter.

Measures real acceptance length (tau) and end-to-end speedup of speculative
decoding with the trained drafter against sequential autoregressive decoding,
under three verification modes:
  greedy      - drafter argmax vs target argmax, exact match (POC-comparable).
  gumbel      - position-indexed Gumbel top-k sampling, exact match.
  stochastic  - Leviathan rejection sampling (distribution-lossless).

Run:  python eval_real_drafter.py --ckpt checkpoints/latest.pt
"""
import argparse
import gc
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "LlamaGen"))

from autoregressive.models.gpt import GPT_models  # noqa: E402
from dflash_visual_drafter import (  # noqa: E402
    RawHiddenCapture, VisualDFlashDrafter, cfg_combine, spec_generate_real, topk_filter,
)

NUM_TOKENS = 576


def cuda_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def make_gumbel(num_tokens, vocab, seed, device):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    u = torch.rand(num_tokens, vocab, generator=g)
    return (-torch.log(-torch.log(u + 1e-20) + 1e-20)).to(device)


@torch.no_grad()
def baseline_generate(model, class_idx, cfg_scale, mode, temperature, top_k,
                      gumbel, seed, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed * 6151 + class_idx)
    cond = torch.tensor([class_idx], device=device)
    cond_combined = torch.cat([cond, torch.full_like(cond, model.num_classes)])
    with torch.device(device):
        model.setup_caches(max_batch_size=2, max_seq_length=1 + NUM_TOKENS,
                           dtype=model.tok_embeddings.weight.dtype)

    def choose(lcfg, i):
        if mode == "greedy":
            return lcfg.argmax(dim=-1)
        if mode == "gumbel":
            l = topk_filter(lcfg / max(temperature, 1e-5), top_k)
            return (l + gumbel[i: i + 1].unsqueeze(0)).argmax(dim=-1)
        p = torch.softmax(topk_filter(lcfg[0, -1] / max(temperature, 1e-5), top_k), dim=-1)
        return torch.multinomial(p, 1, generator=gen).view(1, 1)

    t0 = cuda_time()
    input_pos = torch.arange(0, 1, device=device)
    logits, _ = model(None, cond_combined, input_pos)
    next_token = choose(cfg_combine(logits[:, -1:], cfg_scale), 0)
    seq = torch.empty(NUM_TOKENS, dtype=torch.long, device=device)
    seq[0] = next_token[0, 0]
    input_pos = torch.tensor([1], device=device, dtype=torch.int)
    for i in range(1, NUM_TOKENS):
        x = next_token.view(1, 1)
        logits, _ = model(torch.cat([x, x]), cond_idx=None, input_pos=input_pos)
        next_token = choose(cfg_combine(logits[:, -1:], cfg_scale), i)
        seq[i] = next_token[0, 0]
        input_pos += 1
    return seq, cuda_time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "latest.pt"))
    p.add_argument("--gpt-model", default="GPT-3B")
    p.add_argument("--gpt-ckpt", default=os.path.join(ROOT, "LlamaGen", "pretrained_models", "c2i_3B_384.pt"))
    p.add_argument("--codebook-size", type=int, default=16384)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=2000)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--classes", type=int, nargs="+",
                   default=[207, 360, 387, 974, 88, 979, 417, 279])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--modes", type=str, nargs="+",
                   default=["greedy", "gumbel", "stochastic"])
    p.add_argument("--out", default=os.path.join(ROOT, "results_real"))
    p.add_argument("--config", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--pretrained", default=None)
    p.add_argument("--array-id", type=int, default=0)
    p.add_argument("--array-size", type=int, default=1)
    args = p.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        assert cfg["task"] == "c2i", "use eval_real_drafter_t2i.py for t2i"
        tgt, smp, ev = cfg["target"], cfg["sampling"], cfg["eval"]
        if args.pretrained:
            args.gpt_ckpt = os.path.join(args.pretrained, tgt["gpt_ckpt_rel"])
        args.gpt_model = tgt["gpt_model"]
        args.codebook_size = tgt["codebook_size"]
        args.cfg_scale = smp["cfg_scale"]
        args.temperature = smp["temperature"]
        args.top_k = smp["top_k"]
        args.classes = ev["classes"][args.array_id::args.array_size]
        args.seeds = ev["seeds"]
        args.modes = ev["modes"]
        args.block_size = cfg["drafter"]["block_size"]
        if args.run_dir:
            args.ckpt = os.path.join(args.run_dir, "checkpoints", "latest.pt")
            os.makedirs(os.path.join(args.run_dir, "results"), exist_ok=True)
            args.out = os.path.join(args.run_dir, "results",
                                    f"results_part{args.array_id:02d}")

    device = "cuda"
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    print("Loading target ...", flush=True)
    target = GPT_models[args.gpt_model](vocab_size=args.codebook_size, block_size=NUM_TOKENS,
                                        num_classes=1000, cls_token_num=1, model_type="c2i")
    ckpt = torch.load(args.gpt_ckpt, map_location="cpu", weights_only=True)
    target.load_state_dict(ckpt if "model" not in ckpt else ckpt["model"], strict=False)
    del ckpt
    gc.collect()
    target.to(device=device, dtype=torch.bfloat16).eval()
    torch.cuda.empty_cache()

    print("Loading drafter ...", flush=True)
    st = torch.load(args.ckpt, map_location=device, weights_only=False)
    targs = st.get("args", {})
    drafter = VisualDFlashDrafter(dim=target.config.dim, n_head=target.config.n_head,
                                  num_layers=targs.get("num_layers", 5),
                                  num_features=targs.get("num_features", 5),
                                  block_size=args.block_size)
    drafter.load_state_dict(st["model"])
    drafter.to(device=device, dtype=torch.bfloat16).eval()
    capture = RawHiddenCapture(target, targs.get("num_features", 5))
    print(f"drafter step {st.get('step')} | feature layers {capture.layer_ids}", flush=True)

    # warmup
    spec_generate_real(target, drafter, capture, args.classes[0], NUM_TOKENS,
                       args.cfg_scale, args.block_size, "greedy", device=device)
    baseline_generate(target, args.classes[0], args.cfg_scale, "greedy",
                      args.temperature, args.top_k, None, 0, device)

    results = {"ckpt_step": st.get("step"), "modes": {}}
    for mode in args.modes:
        print(f"\n=== mode: {mode} ===", flush=True)
        base_times, runs = [], []
        for cls in args.classes:
            gum = (make_gumbel(NUM_TOKENS, args.codebook_size, 1000 + cls, device)
                   if mode == "gumbel" else None)
            _, bt = baseline_generate(target, cls, args.cfg_scale, mode,
                                      args.temperature, args.top_k, gum, 0, device)
            base_times.append(bt)
            for seed in (args.seeds if mode != "greedy" else args.seeds[:1]):
                _, stt = spec_generate_real(
                    target, drafter, capture, cls, NUM_TOKENS, args.cfg_scale,
                    args.block_size, mode, args.temperature, args.top_k,
                    gum, seed, device)
                runs.append(stt)
            print(f"  class {cls:4d}: base {bt:.2f}s | spec {runs[-1]['total_s']:.2f}s | "
                  f"tau {runs[-1]['tau']:.2f}", flush=True)
        mean = lambda xs: sum(xs) / len(xs)
        row = {
            "baseline_s": mean(base_times),
            "spec_s": mean([r["total_s"] for r in runs]),
            "draft_s": mean([r["draft_s"] for r in runs]),
            "verify_s": mean([r["verify_s"] for r in runs]),
            "tau": mean([r["tau"] for r in runs]),
            "speedup": mean(base_times) / mean([r["total_s"] for r in runs]),
        }
        results["modes"][mode] = row
        print(f"  => tau {row['tau']:.2f}/{args.block_size} | "
              f"{row['baseline_s']:.2f}s -> {row['spec_s']:.2f}s | "
              f"speedup {row['speedup']:.2f}x", flush=True)

    lines = ["# Real Visual DFlash - Trained Drafter Results", "",
             f"Drafter checkpoint step {st.get('step')} | LlamaGen-3B bf16 | "
             f"CFG {args.cfg_scale} | B={args.block_size} | "
             f"{len(args.classes)} classes x {len(args.seeds)} seeds", "",
             "| Mode | tau (/16) | Baseline (s) | Spec (s) | Draft (s) | Speedup |",
             "|---|---|---|---|---|---|"]
    for mode, r in results["modes"].items():
        lines.append(f"| {mode} | {r['tau']:.2f} | {r['baseline_s']:.2f} | "
                     f"{r['spec_s']:.2f} | {r['draft_s']:.2f} | **{r['speedup']:.2f}x** |")
    report = "\n".join(lines)
    print("\n" + report, flush=True)
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write(report)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.out}.md / .json", flush=True)


if __name__ == "__main__":
    main()
