"""
Debug-mode diagnostic for the trained DFlash t2i drafter.

Loads the trained drafter checkpoint, the LlamaGen target, and a tiny batch of
training data, then runs ONE training-style forward pass. Emits NDJSON logs
(also to stdout) for every hypothesis we're investigating around the low
acc1=0.21 / tau_proxy=1.25 result.

Hypotheses:
  H_A  Double RMSNorm in train path causes train/eval mismatch.
  H_B  fused_ctx is degenerate / drafter doesn't actually use context.
  H_C  anchor_emb (frozen target embedding) vs mask_emb scale mismatch.
  H_D  RoPE indices off by cls_token_num=120 for t2i.
  H_F  Anchor / label / attn_mask index alignment off-by-one.

Usage (called from cluster/lib/debug_drafter.sh):
  python cluster/lib/debug_drafter.py --config <cfg> --run-dir <run> --pretrained <p>
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT_LIB = os.path.dirname(os.path.abspath(__file__))      # .../cluster/lib
REPO = os.path.dirname(os.path.dirname(ROOT_LIB))           # repo root
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "LlamaGen"))

from dflash_visual_drafter import (  # noqa: E402
    RawHiddenCapture, VisualDFlashDrafter,
)
from train_drafter_t2i import (  # noqa: E402
    build_attn_mask, extract_features_t2i, load_target,
)

LOG_PATH = os.path.join(REPO, "debug-a22afb.log")
SESSION_ID = "a22afb"


# #region agent log
def _to_jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().float().cpu().tolist()
    return str(o)


def dlog(hyp, location, message, data):
    rec = {
        "sessionId": SESSION_ID,
        "runId": "diag_t2i_drafter",
        "hypothesisId": hyp,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    line = json.dumps(rec, default=_to_jsonable)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print("DLOG " + line, flush=True)


def t_stats(x):
    """Common summary stats for a tensor."""
    x = x.detach().float()
    return {
        "shape": list(x.shape),
        "mean": x.mean().item(),
        "std": x.std().item(),
        "rms": x.pow(2).mean().sqrt().item(),
        "min": x.min().item(),
        "max": x.max().item(),
        "absmean": x.abs().mean().item(),
    }
# #endregion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--pretrained", required=True)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--anchors", type=int, default=4)
    args = p.parse_args()

    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["task"] == "t2i"

    tgt, dr = cfg["target"], cfg["drafter"]
    latent = tgt["image_size"] // tgt["downsample_size"]
    NUM_TOKENS = latent ** 2
    BLOCK = dr["block_size"]
    cls_tok = tgt["cls_token_num"]
    device = "cuda"
    torch.manual_seed(0)

    print(f"[debug] loading target {tgt['gpt_model']} cls_tok={cls_tok} N={NUM_TOKENS}", flush=True)
    target = load_target(tgt["gpt_model"], os.path.join(args.pretrained, tgt["gpt_ckpt_rel"]),
                         NUM_TOKENS, cls_tok, device)
    capture = RawHiddenCapture(target, dr["num_features"])
    print(f"[debug] feature layers {capture.layer_ids}", flush=True)

    drafter = VisualDFlashDrafter(dim=target.config.dim, n_head=target.config.n_head,
                                  num_layers=dr["num_layers"],
                                  num_features=dr["num_features"],
                                  block_size=BLOCK).to(device=device, dtype=torch.bfloat16)
    ckpt_path = os.path.join(args.run_dir, "checkpoints", "latest.pt")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    drafter.load_state_dict(state["model"])
    drafter.eval()
    print(f"[debug] drafter ckpt loaded step={state.get('step', '?')}", flush=True)

    # ---- tiny batch ----
    shards = sorted(glob.glob(os.path.join(args.run_dir, "data", "shard_*.npz")))
    assert shards, "no token shards found"
    z = np.load(shards[0])
    toks = torch.from_numpy(z["tokens"][:args.batch].astype(np.int64)).to(device)
    pids = z["prompt_ids"][:args.batch]
    z5 = np.load(os.path.join(args.run_dir, "data", "t5_features.npz"))
    t5 = torch.from_numpy(z5["feats"][pids]).to(torch.bfloat16).to(device)
    B = toks.shape[0]
    print(f"[debug] batch toks={tuple(toks.shape)} t5={tuple(t5.shape)}", flush=True)

    # ---- run training-style forward ----
    raw = extract_features_t2i(target, capture, toks, t5, NUM_TOKENS, cls_tok, device)
    fused_ctx = drafter.fuse(raw)

    # ------------ H_B: feature informativeness ------------
    per_layer_rms = []
    D = target.config.dim
    for i in range(dr["num_features"]):
        per_layer_rms.append(raw[:, :, i * D:(i + 1) * D].float().pow(2).mean().sqrt().item())
    pos_var = fused_ctx[0].float().var(dim=-1).cpu()
    dlog("H_B", "debug_drafter.py:fused_ctx",
         "raw and fused context stats; per-layer RMS; per-position variance head/tail",
         {
             "raw_stats": t_stats(raw),
             "raw_per_layer_rms": per_layer_rms,
             "fused_ctx_stats": t_stats(fused_ctx),
             "fused_pos_var_first5": pos_var[:5].tolist(),
             "fused_pos_var_last5": pos_var[-5:].tolist(),
         })

    # ---- pick deterministic anchors for the rest of the diagnostics ----
    ctx_len = NUM_TOKENS - 1
    anchors = torch.tensor([
        [50, 200, 500, 900],
        [10, 100, 400, 800],
        [25, 150, 600, 950],
        [75, 300, 700, 990],
    ], device=device, dtype=torch.long)[:B, :args.anchors]
    anchor_tok = toks.gather(1, anchors)
    anchor_emb = target.tok_embeddings(anchor_tok)
    mask_emb = drafter.mask_embedding.to(anchor_emb.dtype)

    # ------------ H_C: anchor vs mask scale ------------
    dlog("H_C", "debug_drafter.py:block_emb",
         "anchor vs mask embedding scale",
         {
             "anchor_emb_stats": t_stats(anchor_emb),
             "mask_emb_stats": t_stats(mask_emb),
             "ratio_anchor_to_mask_rms": (
                 anchor_emb.float().pow(2).mean().sqrt() /
                 (mask_emb.float().pow(2).mean().sqrt() + 1e-12)).item(),
         })

    block_emb = torch.cat([
        anchor_emb.unsqueeze(2),
        mask_emb.view(1, 1, 1, -1).expand(B, args.anchors, BLOCK - 1, -1)],
        dim=2).reshape(B, args.anchors * BLOCK, -1)

    freqs = target.freqs_cis
    ctx_freqs = freqs[cls_tok: cls_tok + NUM_TOKENS - 1]
    block_pos_img = anchors.unsqueeze(-1) + torch.arange(BLOCK, device=device)
    block_freqs = freqs[(cls_tok + block_pos_img).reshape(B, -1)]
    attn_mask = build_attn_mask(anchors, ctx_len, BLOCK, device)

    # ------------ H_D: RoPE alignment ------------
    p0 = anchors[0, 0].item()
    dlog("H_D", "debug_drafter.py:rope",
         "RoPE freqs alignment for t2i cls_tok=120",
         {
             "freqs_full_shape": list(freqs.shape),
             "cls_tok": cls_tok,
             "ctx_freqs_shape": list(ctx_freqs.shape),
             "ctx_freqs_imgpos0_first8": ctx_freqs[0].float().flatten().cpu().tolist()[:8],
             "target_freqs_at_cls_first8": freqs[cls_tok].float().flatten().cpu().tolist()[:8],
             "ctx0_equal_target_at_cls": bool(torch.allclose(
                 ctx_freqs[0].float(), freqs[cls_tok].float())),
             "block_freqs_b0_a0_row0_first8": block_freqs[0, 0].float().flatten().cpu().tolist()[:8],
             "block_freqs_b0_a0_row0_target_idx": cls_tok + p0,
             "target_freqs_at_p0_first8": freqs[cls_tok + p0].float().flatten().cpu().tolist()[:8],
             "block_freqs_b0_a0_row0_eq_target": bool(torch.allclose(
                 block_freqs[0, 0].float(), freqs[cls_tok + p0].float())),
         })

    # ------------ H_F: anchor/label index alignment ------------
    lab_idx = block_pos_img[:, :, 1:]
    labels_blk = toks.gather(1, lab_idx.reshape(B, -1)).view(B, args.anchors, BLOCK - 1)
    dlog("H_F", "debug_drafter.py:labels",
         "anchor + label index alignment sanity",
         {
             "anchors_b0": anchors[0].cpu().tolist(),
             "block_pos_img_b0_a0": block_pos_img[0, 0].cpu().tolist(),
             "lab_idx_b0_a0": lab_idx[0, 0].cpu().tolist(),
             "anchor_tok_b0": anchor_tok[0].cpu().tolist(),
             "tokens_at_block_pos_b0_a0": toks[0, block_pos_img[0, 0]].cpu().tolist(),
             "tokens_at_lab_idx_b0_a0_eq_labels": bool(
                 (toks[0, lab_idx[0, 0]] == labels_blk[0, 0]).all().item()),
         })

    # ------------ H_F (cont): attn_mask True counts ------------
    ctx_true = [int(attn_mask[0, 0, a * BLOCK, :ctx_len].sum().item()) for a in range(args.anchors)]
    self_true = [int(attn_mask[0, 0, a * BLOCK, ctx_len:].sum().item()) for a in range(args.anchors)]
    dlog("H_F", "debug_drafter.py:attn_mask",
         "attn_mask True counts per anchor: ctx visible == anchor pos; self visible == BLOCK*anchors? "
         "(self_vis is block-diagonal full, so per-row self True count = BLOCK)",
         {
             "shape": list(attn_mask.shape),
             "ctx_true_counts_b0": ctx_true,
             "expected_ctx_true_counts_b0": anchors[0].cpu().tolist(),
             "self_true_counts_b0": self_true,
             "expected_self_true_count_per_anchor": BLOCK,
         })

    # ------------ Forward pass + H_A double-norm test ------------
    hid_pre = drafter.forward_train(fused_ctx, ctx_freqs, block_emb, block_freqs, attn_mask)
    hid_view = hid_pre.view(B, args.anchors, BLOCK, -1)[:, :, 1:, :]
    logits_train = target.output(drafter.norm(hid_view)).float()    # what TRAIN uses
    logits_eval = target.output(hid_view).float()                   # single-norm (matches inference draft_logits)

    acc_train = (logits_train.argmax(-1) == labels_blk).float().mean(dim=(0, 1)).cpu().tolist()
    acc_eval = (logits_eval.argmax(-1) == labels_blk).float().mean(dim=(0, 1)).cpu().tolist()

    dlog("H_A", "debug_drafter.py:double_norm",
         "Train path (double norm) vs eval path (single norm) - acc1..acc15",
         {
             "hid_pre_stats": t_stats(hid_pre),
             "drafter_norm_renorm_stats": t_stats(drafter.norm(hid_view)),
             "drafter_norm_weight_stats": t_stats(drafter.norm.weight),
             "acc_per_pos_TRAIN_doubled": acc_train,
             "acc_per_pos_EVAL_single":  acc_eval,
             "acc1_train_vs_eval": [acc_train[0], acc_eval[0]],
         })

    # ------------ H_B (cont): context-ablation, zero out fused_ctx ------------
    fused_zero = torch.zeros_like(fused_ctx)
    hid_pre_noctx = drafter.forward_train(fused_zero, ctx_freqs, block_emb, block_freqs, attn_mask)
    hid_view_noctx = hid_pre_noctx.view(B, args.anchors, BLOCK, -1)[:, :, 1:, :]
    # match training regime so we compare apples to apples vs acc_train above
    logits_noctx = target.output(drafter.norm(hid_view_noctx)).float()
    acc_noctx = (logits_noctx.argmax(-1) == labels_blk).float().mean(dim=(0, 1)).cpu().tolist()
    dlog("H_B", "debug_drafter.py:noctx",
         "Context ablation: zero fused_ctx; if acc unchanged => drafter ignores ctx",
         {
             "acc_per_pos_with_zero_ctx": acc_noctx,
             "acc1_real_vs_zeroctx": [acc_train[0], acc_noctx[0]],
             "delta_acc1_from_ctx": acc_train[0] - acc_noctx[0],
             "delta_acc8_from_ctx": acc_train[7] - acc_noctx[7],
         })

    # ------------ H_J: drafter parameter health ------------
    # Drafter init: linear weights N(0, 0.02), residual-scaled wo/w2 N(0, 0.02/sqrt(2*5)=0.00632),
    # RMSNorm weights init=1.0, mask_embedding N(0, 0.02). Bias-less, so no biases.
    # If a param's stats are exactly the init's stats, training didn't move it.
    param_stats = {}
    for name, pp in drafter.named_parameters():
        s = pp.detach().float()
        param_stats[name] = {
            "shape": list(s.shape),
            "rms": s.pow(2).mean().sqrt().item(),
            "mean": s.mean().item(),
            "std": s.std().item(),
            "min": s.min().item(),
            "max": s.max().item(),
        }
    dlog("H_J", "debug_drafter.py:param_health",
         "drafter param stats; check if training actually moved weights vs init",
         {"params": param_stats})

    # ------------ H_M: drafter argmax vs TARGET argmax (the right metric) ------------
    # Re-run target to also grab its full-sequence logits (not captured by hooks).
    idx_seq = toks[:, :NUM_TOKENS - 1].to(device)
    seq_len_full = cls_tok + idx_seq.shape[1]
    ip_full = torch.arange(seq_len_full, device=device)
    with torch.no_grad():
        target_logits_all, _ = target(idx_seq, cond_idx=t5, input_pos=ip_full, targets=None)
    # target_logits_all has shape (B, cls_tok + N-1, V). Position k of this tensor is the
    # NEXT-TOKEN prediction for what comes at sequence position k+1.
    # Image position p corresponds to sequence position cls_tok + p.
    # To compare drafter's prediction at image position p+1 (i.e. tokens[p+1]), use
    # target's logits at sequence position cls_tok + p, i.e. block_pos_img[:, :, :-1] + cls_tok.
    seq_pos = cls_tok + block_pos_img[:, :, :-1]                    # (B, A, 15)
    B_, A_, K_ = seq_pos.shape
    V = target_logits_all.shape[-1]
    target_logits_block = torch.gather(
        target_logits_all, dim=1,
        index=seq_pos.reshape(B_, -1).unsqueeze(-1).expand(-1, -1, V),
    ).view(B_, A_, K_, V).float()                                   # (B, A, 15, V)

    target_argmax = target_logits_block.argmax(-1)                  # (B, A, 15)
    drafter_argmax = logits_train.argmax(-1)                        # (B, A, 15), train regime
    drafter_argmax_eval = logits_eval.argmax(-1)                    # eval regime

    agree_target_train = (drafter_argmax == target_argmax).float().mean(dim=(0, 1)).cpu().tolist()
    agree_target_eval = (drafter_argmax_eval == target_argmax).float().mean(dim=(0, 1)).cpu().tolist()

    target_top5 = target_logits_block.topk(5, dim=-1).indices       # (B, A, 15, 5)
    in_top5 = (drafter_argmax.unsqueeze(-1) == target_top5).any(-1).float().mean(dim=(0, 1)).cpu().tolist()
    target_top20 = target_logits_block.topk(20, dim=-1).indices
    in_top20 = (drafter_argmax.unsqueeze(-1) == target_top20).any(-1).float().mean(dim=(0, 1)).cpu().tolist()

    target_argmax_eq_data = (target_argmax == labels_blk).float().mean(dim=(0, 1)).cpu().tolist()
    target_probs = F.softmax(target_logits_block, dim=-1)
    target_top1_prob = target_probs.gather(-1, target_argmax.unsqueeze(-1)).squeeze(-1)
    target_top1_prob_mean = target_top1_prob.mean(dim=(0, 1)).cpu().tolist()
    target_entropy_t = -(target_probs * target_probs.clamp_min(1e-30).log()).sum(-1).mean(dim=(0, 1))
    target_entropy = target_entropy_t.cpu().tolist()

    dlog("H_M", "debug_drafter.py:target_argmax_metric",
         "drafter top-1 vs TARGET top-1 (right metric for spec decoding) + target's distribution stats",
         {
             "acc_vs_DATA_SAMPLE": acc_train,
             "acc_vs_TARGET_ARGMAX": agree_target_train,
             "acc_vs_TARGET_ARGMAX_eval_regime": agree_target_eval,
             "drafter_top1_in_target_TOP5": in_top5,
             "drafter_top1_in_target_TOP20": in_top20,
             "target_argmax_matches_DATA_sample": target_argmax_eq_data,
             "target_top1_prob_per_pos": target_top1_prob_mean,
             "target_entropy_per_pos": target_entropy,
         })

    print("[debug] done; log at " + LOG_PATH, flush=True)


if __name__ == "__main__":
    main()
