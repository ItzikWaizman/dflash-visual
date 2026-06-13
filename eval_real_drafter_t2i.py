"""
Phase 3 (t2i): Evaluate trained DFlash drafter for LlamaGen-T2I targets.

Measures real tau and end-to-end speedup against sequential AR decoding,
under three verification modes (greedy / gumbel / stochastic).

Designed for SLURM job arrays: each task evaluates a disjoint prompt
subset, results merged at the end.
"""
import argparse
import gc
import glob
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "LlamaGen"))

from autoregressive.models.gpt import GPT_models  # noqa: E402
from dflash_visual_drafter import (  # noqa: E402
    RawHiddenCapture, VisualDFlashDrafter, cfg_combine, topk_filter,
)


def cuda_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def make_gumbel(L, V, seed, device):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    u = torch.rand(L, V, generator=g)
    return (-torch.log(-torch.log(u + 1e-20) + 1e-20)).to(device)


@torch.no_grad()
def prefill_t2i(model, c_embed, cls_token_num, device):
    """Replicate LlamaGen's prefill: cls_embedding(text) -> transformer ->
    logits at last cls position. Returns (logits_last, cfg-combined-already-False)."""
    b = c_embed.shape[0]
    uncond = torch.cat([torch.zeros(b // 2, dtype=torch.long, device=device),
                        torch.ones(b // 2, dtype=torch.long, device=device)])
    cond_emb = model.cls_embedding(c_embed, train=False, force_drop_ids=uncond)
    h = model.tok_dropout(cond_emb[:, :cls_token_num])
    input_pos = torch.arange(0, cls_token_num, device=device)
    mask = model.causal_mask[:b, None, input_pos]
    freqs_cis = model.freqs_cis[input_pos]
    for layer in model.layers:
        h = layer(h, freqs_cis, input_pos, mask)
    return model.output(model.norm(h)).float()


@torch.no_grad()
def baseline_t2i(model, c_embed, num_tokens, cls_token_num, cfg_scale,
                  mode, temperature, top_k, gumbel, seed, device):
    """One image baseline (batch=2 for CFG)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    with torch.device(device):
        model.setup_caches(max_batch_size=2, max_seq_length=cls_token_num + num_tokens,
                           dtype=torch.bfloat16)
    c_combined = torch.cat([c_embed, c_embed], dim=0)

    def choose(lcfg, i):
        if mode == "greedy":
            return lcfg.argmax(dim=-1)
        if mode == "gumbel":
            l = topk_filter(lcfg / max(temperature, 1e-5), top_k)
            return (l + gumbel[i: i + 1].unsqueeze(0)).argmax(dim=-1)
        p = torch.softmax(topk_filter(lcfg[0, -1] / max(temperature, 1e-5), top_k), dim=-1)
        return torch.multinomial(p, 1, generator=gen).view(1, 1)

    t0 = cuda_time()
    logits = prefill_t2i(model, c_combined, cls_token_num, device)
    next_token = choose(cfg_combine(logits[:, -1:], cfg_scale), 0)
    seq = torch.empty(num_tokens, dtype=torch.long, device=device)
    seq[0] = next_token[0, 0]
    input_pos = torch.tensor([cls_token_num], device=device, dtype=torch.int)
    for i in range(1, num_tokens):
        x = next_token.view(1, 1)
        logits, _ = model(torch.cat([x, x]), cond_idx=None, input_pos=input_pos)
        next_token = choose(cfg_combine(logits[:, -1:], cfg_scale), i)
        seq[i] = next_token[0, 0]
        input_pos += 1
    return seq, cuda_time() - t0


@torch.no_grad()
def spec_t2i(target, drafter, capture, c_embed, num_tokens, cls_token_num,
              cfg_scale, block_size, mode, temperature, top_k, gumbel,
              seed, device):
    """T2I version of spec_generate_real. Drafter sees IMAGE-position features
    only; freqs_cis lookups offset by cls_token_num."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    with torch.device(device):
        target.setup_caches(max_batch_size=2, max_seq_length=cls_token_num + num_tokens,
                            dtype=torch.bfloat16)
    drafter.reset_cache(cls_offset=cls_token_num)
    freqs = target.freqs_cis.to(device)
    c_combined = torch.cat([c_embed, c_embed], dim=0)

    def tchoice(lcfg, pos0):
        if mode == "greedy":
            return lcfg.argmax(dim=-1)
        l = topk_filter(lcfg / max(temperature, 1e-5), top_k)
        L = lcfg.shape[1]
        return (l + gumbel[pos0: pos0 + L].unsqueeze(0)).argmax(dim=-1)

    t0 = cuda_time()
    draft_time = 0.0
    # prefill with text
    logits = prefill_t2i(target, c_combined, cls_token_num, device)
    lcfg = cfg_combine(logits, cfg_scale)
    if mode == "stochastic":
        p = torch.softmax(topk_filter(lcfg[0, -1] / max(temperature, 1e-5), top_k), dim=-1)
        first = torch.multinomial(p, 1, generator=gen).view(1, 1)
    else:
        first = tchoice(lcfg[:, -1:], 0)

    seq = torch.empty(num_tokens, dtype=torch.long, device=device)
    seq[0] = first[0, 0]
    n = 1
    accept_lens = []

    # Run the target on the first sampled token to get its image-pos features
    input_pos = torch.tensor([cls_token_num], device=device, dtype=torch.int)
    target(torch.cat([first, first]), cond_idx=None, input_pos=input_pos)
    img_raw_0 = capture.concat()                                 # (1, 1, 5D)
    drafter.append_context(drafter.fuse(img_raw_0), freqs)

    while n < num_tokens:
        draft_len = min(block_size, num_tokens - n)

        td = cuda_time()
        anchor_emb = target.tok_embeddings(seq[n - 1].view(1, 1)).to(target.tok_embeddings.weight.dtype)
        block = torch.empty(1, draft_len, dtype=torch.long, device=device)
        block[0, 0] = seq[n - 1]
        q_probs = None
        if draft_len > 1:
            # IMAGE position n in target-coord = cls_token_num + n
            dlogits = drafter.draft_logits(anchor_emb, cls_token_num + n,
                                             draft_len, freqs, target.output).float()
            if mode == "stochastic":
                q_probs = torch.softmax(dlogits[0] / max(temperature, 1e-5), dim=-1)
                block[0, 1:] = torch.multinomial(q_probs, 1, generator=gen).view(-1)
            else:
                block[0, 1:] = dlogits.argmax(dim=-1)[0]
        draft_time += cuda_time() - td

        # verify
        input_pos = torch.arange(cls_token_num + n, cls_token_num + n + draft_len,
                                  device=device, dtype=torch.int)
        logits, _ = target(torch.cat([block, block]), cond_idx=None, input_pos=input_pos)
        lcfg = cfg_combine(logits, cfg_scale)

        if mode == "stochastic":
            p_probs = torch.softmax(topk_filter(lcfg[0] / max(temperature, 1e-5), top_k), dim=-1)
            accepted = 0
            corrected = None
            for j in range(draft_len - 1):
                x = block[0, j + 1]
                pj, qj = p_probs[j, x], q_probs[j, x]
                if torch.rand((), device=device, generator=gen) * qj <= pj:
                    accepted += 1
                else:
                    resid = torch.clamp(p_probs[j] - q_probs[j], min=0)
                    resid = resid / resid.sum()
                    corrected = torch.multinomial(resid, 1, generator=gen).view(())
                    break
            if corrected is None:
                corrected = torch.multinomial(p_probs[draft_len - 1], 1, generator=gen).view(())
            seq[n: n + accepted] = block[0, 1: 1 + accepted]
            seq[n + accepted] = corrected
        else:
            preds = tchoice(lcfg, n)
            if draft_len > 1:
                matches = (block[0, 1:] == preds[0, :-1]).int()
                accepted = int(torch.cumprod(matches, dim=0).sum().item())
            else:
                accepted = 0
            seq[n: n + accepted] = block[0, 1: 1 + accepted]
            seq[n + accepted] = preds[0, accepted]

        new = accepted + 1
        accept_lens.append(new)
        raw_block = capture.concat()                              # (1, draft_len, 5D)
        drafter.append_context(drafter.fuse(raw_block)[:, :new, :], freqs)
        n += new

    el = cuda_time() - t0
    return seq, {"total_s": el, "draft_s": draft_time, "verify_s": el - draft_time,
                  "tau": sum(accept_lens) / len(accept_lens),
                  "steps": len(accept_lens), "acceptance_lengths": accept_lens}


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
    torch.cuda.empty_cache()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--pretrained", required=True)
    p.add_argument("--array-id", type=int, default=0)
    p.add_argument("--array-size", type=int, default=1)
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["task"] == "t2i"
    tgt, smp, ev, dr = cfg["target"], cfg["sampling"], cfg["eval"], cfg["drafter"]
    latent = tgt["image_size"] // tgt["downsample_size"]
    NUM_TOKENS = latent ** 2
    cls_tok = tgt["cls_token_num"]
    BLOCK = dr["block_size"]

    device = "cuda"
    print("[t2i eval] loading target", flush=True)
    target = load_target(tgt["gpt_model"],
                         os.path.join(args.pretrained, tgt["gpt_ckpt_rel"]),
                         NUM_TOKENS, cls_tok, device)

    ckpt_path = os.path.join(args.run_dir, "checkpoints", "latest.pt")
    print(f"[t2i eval] loading drafter from {ckpt_path}", flush=True)
    st = torch.load(ckpt_path, map_location=device, weights_only=False)
    drafter = VisualDFlashDrafter(dim=target.config.dim, n_head=target.config.n_head,
                                  num_layers=dr["num_layers"],
                                  num_features=dr["num_features"], block_size=BLOCK)
    drafter.load_state_dict(st["model"])
    drafter.to(device=device, dtype=torch.bfloat16).eval()
    capture = RawHiddenCapture(target, dr["num_features"])

    # ---- prepare eval prompts ----
    data_root = os.environ.get("DFLASH_DATA",
                               os.path.dirname(os.path.dirname(args.run_dir)) + "/data")
    eval_prompts_file = os.path.join(data_root, ev["prompts_file_rel"])
    prompts_all = []
    with open(eval_prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            cap = o.get("caption") or o.get("text") or o.get("prompt")
            if cap:
                prompts_all.append(cap)
    np.random.default_rng(0).shuffle(prompts_all)
    prompts_all = prompts_all[: ev["num_prompts"]]
    my_prompts = prompts_all[args.array_id::args.array_size]
    print(f"[t2i eval] {len(my_prompts)} prompts for this task", flush=True)

    # encode them with T5
    from language.t5 import T5Embedder
    t5 = T5Embedder(device=device, local_cache=True,
                    cache_dir=os.path.join(args.pretrained, cfg["text_encoder"]["cache_rel"]),
                    dir_or_name=cfg["text_encoder"]["model"], torch_dtype=torch.bfloat16,
                    model_max_length=cls_tok)
    emb, m = t5.get_text_embeddings(my_prompts)
    m_flip = torch.flip(m, dims=[-1])
    rolled = torch.empty_like(emb)
    for i in range(emb.shape[0]):
        valid = int(m[i].sum().item())
        rolled[i] = torch.cat([emb[i, valid:], emb[i, :valid]], dim=0)
    t5_feats = rolled.to(torch.bfloat16)
    del t5
    gc.collect()
    torch.cuda.empty_cache()

    results = {"ckpt_step": st.get("step"), "modes": {}, "per_prompt": []}
    for mode in ev["modes"]:
        print(f"\n=== mode: {mode} ===", flush=True)
        base_times, runs = [], []
        for i, pmt in enumerate(my_prompts):
            c_e = t5_feats[i: i + 1]
            gum = (make_gumbel(NUM_TOKENS, tgt["codebook_size"], 1000 + i, device)
                   if mode == "gumbel" else None)
            for seed in (ev["seeds"] if mode != "greedy" else ev["seeds"][:1]):
                _, bt = baseline_t2i(target, c_e, NUM_TOKENS, cls_tok,
                                      smp["cfg_scale"], mode, smp["temperature"],
                                      smp["top_k"], gum, seed, device)
                base_times.append(bt)
                _, stt = spec_t2i(target, drafter, capture, c_e, NUM_TOKENS, cls_tok,
                                   smp["cfg_scale"], BLOCK, mode, smp["temperature"],
                                   smp["top_k"], gum, seed, device)
                runs.append(stt)
                print(f"  prompt {i:3d} seed {seed}: base {bt:.2f}s | "
                      f"spec {stt['total_s']:.2f}s | tau {stt['tau']:.2f}", flush=True)
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
        print(f"  => tau {row['tau']:.2f}/{BLOCK} | speedup {row['speedup']:.2f}x", flush=True)

    out_root = os.path.join(args.run_dir, "results")
    os.makedirs(out_root, exist_ok=True)
    out_path = os.path.join(out_root, f"results_part{args.array_id:02d}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[t2i eval] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
