"""
Visual DFlash drafter: a real block-diffusion draft model for LlamaGen.

Faithful port of the DFlash architecture (arXiv:2602.06036) to visual AR:
  - N transformer layers at target width (LlamaGen-3B: dim 3200, 32 heads).
  - Fused target context features (5 hidden layers, concat -> fc -> RMSNorm)
    KV-INJECTED into every layer: ctx features are projected by each layer's
    k/v projections and concatenated in front of the block's own k/v. Ctx keys
    carry the target's 2D RoPE at their absolute positions.
  - Bidirectional attention within the drafted block; learned mask embedding;
    frozen shared token embedding and LM head from the target.

This file also provides:
  - RawHiddenCapture: forward hooks on the target collecting the concatenated
    5-layer hidden states (the raw input to the drafter's fuse()).
  - spec_generate_real(): DFlash-style speculative decoding loop for LlamaGen
    using a trained drafter, with greedy / gumbel / stochastic verification.
"""
import os
import sys
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "LlamaGen"))

from autoregressive.models.gpt import RMSNorm, find_multiple  # noqa: E402


def build_target_layer_ids(num_target_layers: int, num_features: int):
    if num_features == 1:
        return [num_target_layers // 2]
    start, end = 1, num_target_layers - 3
    span = end - start
    return [int(round(start + (i * span) / (num_features - 1))) for i in range(num_features)]


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """x: (B, L, H, hd); freqs: (L, hd//2, 2) or (B, L, hd//2, 2)."""
    xs = x.float().reshape(*x.shape[:-1], -1, 2)
    if freqs.dim() == 3:
        freqs = freqs.unsqueeze(0)  # (1, L, hd//2, 2)
    fr = freqs.unsqueeze(2).to(x.device)  # (B|1, L, 1, hd//2, 2)
    out = torch.stack([
        xs[..., 0] * fr[..., 0] - xs[..., 1] * fr[..., 1],
        xs[..., 1] * fr[..., 0] + xs[..., 0] * fr[..., 1],
    ], dim=-1)
    return out.flatten(3).type_as(x)


class DrafterLayer(nn.Module):
    def __init__(self, dim: int, n_head: int, ffn_hidden: int):
        super().__init__()
        self.n_head, self.head_dim = n_head, dim // n_head
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.w1 = nn.Linear(dim, ffn_hidden, bias=False)
        self.w3 = nn.Linear(dim, ffn_hidden, bias=False)
        self.w2 = nn.Linear(ffn_hidden, dim, bias=False)
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        b, L, _ = t.shape
        return t.view(b, L, self.n_head, self.head_dim)

    def ctx_kv(self, fused_ctx: torch.Tensor, ctx_freqs: torch.Tensor):
        """Project fused target features to RoPE'd K and V for this layer.
        fused_ctx: (B, Lc, D); ctx_freqs: (Lc, hd//2, 2) or (B, Lc, hd//2, 2).
        Returns k, v as (B, H, Lc, hd)."""
        k = apply_rope(self._heads(self.wk(fused_ctx)), ctx_freqs).transpose(1, 2)
        v = self._heads(self.wv(fused_ctx)).transpose(1, 2)
        return k, v

    def forward(self, x: torch.Tensor, block_freqs: torch.Tensor,
                ctx_k: torch.Tensor, ctx_v: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, Lq, D) block hidden states. ctx_k/ctx_v: (B, H, Lc, hd).
        attn_mask: bool (B, 1, Lq, Lc+Lq) or None (= all visible)."""
        b, Lq, _ = x.shape
        h = self.attn_norm(x)
        q = apply_rope(self._heads(self.wq(h)), block_freqs).transpose(1, 2)
        k_self = apply_rope(self._heads(self.wk(h)), block_freqs).transpose(1, 2)
        v_self = self._heads(self.wv(h)).transpose(1, 2)
        k = torch.cat([ctx_k, k_self], dim=2)
        v = torch.cat([ctx_v, v_self], dim=2)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.wo(attn.transpose(1, 2).reshape(b, Lq, -1))
        h = self.ffn_norm(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class VisualDFlashDrafter(nn.Module):
    def __init__(self, dim: int = 3200, n_head: int = 32, num_layers: int = 5,
                 num_features: int = 5, block_size: int = 16):
        super().__init__()
        self.dim, self.block_size = dim, block_size
        self.num_features = num_features
        ffn_hidden = find_multiple(int(2 * 4 * dim / 3), 256)
        self.fc = nn.Linear(num_features * dim, dim, bias=False)
        self.hidden_norm = RMSNorm(dim)
        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        self.layers = nn.ModuleList(
            [DrafterLayer(dim, n_head, ffn_hidden) for _ in range(num_layers)])
        self.norm = RMSNorm(dim)
        self._init_weights(num_layers)
        # inference-time ctx KV cache, one (k, v) per layer
        self._ctx_k: list = [None] * num_layers
        self._ctx_v: list = [None] * num_layers
        self._ctx_len = 0
        self._cls_offset = 0  # set via reset_cache(cls_offset=...) for t2i targets

    def _init_weights(self, num_layers: int):
        std = 0.02
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
        for layer in self.layers:  # residual-scaled init
            nn.init.normal_(layer.wo.weight, mean=0.0, std=std / (2 * num_layers) ** 0.5)
            nn.init.normal_(layer.w2.weight, mean=0.0, std=std / (2 * num_layers) ** 0.5)
        nn.init.normal_(self.mask_embedding, mean=0.0, std=std)

    def fuse(self, raw_cat: torch.Tensor) -> torch.Tensor:
        """raw_cat: (B, L, num_features*D) concatenated target hidden states."""
        return self.hidden_norm(self.fc(raw_cat))

    # ------------------------------------------------------------- training
    def forward_train(self, fused_ctx: torch.Tensor, ctx_freqs: torch.Tensor,
                      block_emb: torch.Tensor, block_freqs: torch.Tensor,
                      attn_mask: torch.Tensor) -> torch.Tensor:
        """fused_ctx (B, Lc, D); block_emb (B, A*S, D); block_freqs (B, A*S, hd//2, 2);
        attn_mask bool (B, 1, A*S, Lc+A*S). Returns final hidden (B, A*S, D)."""
        x = block_emb
        for layer in self.layers:
            ck, cv = layer.ctx_kv(fused_ctx, ctx_freqs)
            x = layer(x, block_freqs, ck, cv, attn_mask)
        return self.norm(x)

    # ------------------------------------------------------------ inference
    def reset_cache(self, cls_offset: int = 0):
        """cls_offset: how many entries in the target's freqs_cis come BEFORE the
        first image position (1 for c2i, 120 for t2i). Must be set before
        any append_context call."""
        self._ctx_k = [None] * len(self.layers)
        self._ctx_v = [None] * len(self.layers)
        self._ctx_len = 0
        self._cls_offset = cls_offset

    @torch.no_grad()
    def append_context(self, fused_ctx: torch.Tensor, freqs_cis: torch.Tensor):
        """fused_ctx: (1, m, D) image-position features for image positions
        [_ctx_len, _ctx_len+m). freqs_cis: full target freqs table; this
        method offsets by self._cls_offset to skip the cls/text slots."""
        m = fused_ctx.shape[1]
        lo = self._cls_offset + self._ctx_len
        pos_freqs = freqs_cis[lo: lo + m]
        for i, layer in enumerate(self.layers):
            k, v = layer.ctx_kv(fused_ctx, pos_freqs)
            if self._ctx_k[i] is None:
                self._ctx_k[i], self._ctx_v[i] = k, v
            else:
                self._ctx_k[i] = torch.cat([self._ctx_k[i], k], dim=2)
                self._ctx_v[i] = torch.cat([self._ctx_v[i], v], dim=2)
        self._ctx_len += m

    @torch.no_grad()
    def draft_logits(self, anchor_emb: torch.Tensor, start_pos: int, draft_len: int,
                     freqs_cis: torch.Tensor, lm_head: nn.Module) -> torch.Tensor:
        """anchor_emb (1,1,D); block at absolute positions [start_pos, start_pos+draft_len).
        Returns logits (1, draft_len-1, V) for the masked positions."""
        x = torch.cat([anchor_emb,
                       self.mask_embedding.expand(1, draft_len - 1, -1).to(anchor_emb.dtype)],
                      dim=1)
        block_freqs = freqs_cis[start_pos: start_pos + draft_len]
        for i, layer in enumerate(self.layers):
            x = layer(x, block_freqs, self._ctx_k[i], self._ctx_v[i], attn_mask=None)
        return lm_head(self.norm(x[:, 1:, :]))


class RawHiddenCapture:
    """Hooks on selected target layers; returns concat (B, L, 5D) on demand."""

    def __init__(self, target: nn.Module, num_features: int):
        self.layer_ids = build_target_layer_ids(target.n_layer, num_features)
        self._captured: dict[int, torch.Tensor] = {}
        self._handles = [
            target.layers[lid].register_forward_hook(self._make_hook(lid))
            for lid in self.layer_ids
        ]

    def _make_hook(self, lid: int):
        def hook(_m, _i, output):
            self._captured[lid] = output
        return hook

    def concat(self, cond_only: bool = True) -> torch.Tensor:
        cat = torch.cat([self._captured[lid] for lid in self.layer_ids], dim=-1)
        if cond_only and cat.shape[0] % 2 == 0:
            cat = cat[: cat.shape[0] // 2]
        return cat

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Real speculative decoding loop
# ---------------------------------------------------------------------------

def _cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def cfg_combine(logits: torch.Tensor, cfg_scale: float) -> torch.Tensor:
    cond, uncond = torch.split(logits, logits.shape[0] // 2, dim=0)
    return uncond + (cond - uncond) * cfg_scale


def topk_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return logits
    thresh = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return torch.where(logits < thresh, torch.full_like(logits, -float("inf")), logits)


@torch.no_grad()
def spec_generate_real(target: nn.Module, drafter: VisualDFlashDrafter,
                       capture: RawHiddenCapture, class_idx: int, num_tokens: int,
                       cfg_scale: float, block_size: int, mode: str,
                       temperature: float = 1.0, top_k: int = 2000,
                       gumbel: Optional[torch.Tensor] = None,
                       seed: int = 0, device: str = "cuda"):
    """mode: 'greedy' | 'gumbel' | 'stochastic'.

    greedy:     draft = drafter argmax; verify = target argmax; exact-match accept.
    gumbel:     draft = drafter argmax; verify = position-indexed Gumbel top-k
                sample; exact-match accept (needs `gumbel` (num_tokens, V)).
    stochastic: draft sampled from drafter distribution q; Leviathan verification
                accept w.p. min(1, p/q), resample from (p-q)+ on rejection.
                Output is distribution-lossless w.r.t. the target.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed * 6151 + class_idx)
    dtype = target.tok_embeddings.weight.dtype
    cond = torch.tensor([class_idx], device=device)
    cond_combined = torch.cat([cond, torch.full_like(cond, target.num_classes)])
    with torch.device(device):
        target.setup_caches(max_batch_size=2, max_seq_length=1 + num_tokens, dtype=dtype)
    drafter.reset_cache()
    freqs = target.freqs_cis.to(device)  # (1+num_tokens, hd//2, 2)

    t0 = _cuda_time()
    draft_time = 0.0

    def target_choice(logits_cfg: torch.Tensor, pos0: int):
        """logits_cfg (1, L, V) -> chosen tokens (1, L) under the active mode."""
        if mode == "greedy":
            return logits_cfg.argmax(dim=-1)
        if mode == "gumbel":
            l = topk_filter(logits_cfg / max(temperature, 1e-5), top_k)
            L = logits_cfg.shape[1]
            return (l + gumbel[pos0: pos0 + L].unsqueeze(0)).argmax(dim=-1)
        raise RuntimeError(mode)

    # prefill: class token at position 0
    input_pos = torch.arange(0, 1, device=device)
    logits, _ = target(None, cond_combined, input_pos)
    lcfg = cfg_combine(logits, cfg_scale)
    if mode == "stochastic":
        p = torch.softmax(topk_filter(lcfg[0, -1] / max(temperature, 1e-5), top_k), dim=-1)
        first = torch.multinomial(p, 1, generator=gen).view(1, 1)
    else:
        first = target_choice(lcfg[:, -1:], 0)
    drafter.append_context(drafter.fuse(capture.concat()), freqs)

    seq = torch.empty(num_tokens, dtype=torch.long, device=device)
    seq[0] = first[0, 0]
    n = 1
    acceptance_lengths = []

    while n < num_tokens:
        draft_len = min(block_size, num_tokens - n)

        td = _cuda_time()
        anchor_emb = target.tok_embeddings(seq[n - 1].view(1, 1)).to(dtype)
        block = torch.empty(1, draft_len, dtype=torch.long, device=device)
        block[0, 0] = seq[n - 1]
        q_probs = None
        if draft_len > 1:
            dlogits = drafter.draft_logits(anchor_emb, n, draft_len, freqs, target.output)
            dlogits = dlogits.float()
            if mode == "stochastic":
                q_probs = torch.softmax(dlogits[0] / max(temperature, 1e-5), dim=-1)
                block[0, 1:] = torch.multinomial(q_probs, 1, generator=gen).view(-1)
            else:
                block[0, 1:] = dlogits.argmax(dim=-1)[0]
        draft_time += _cuda_time() - td

        # parallel verification forward (positions n .. n+draft_len-1)
        input_pos = torch.arange(n, n + draft_len, device=device, dtype=torch.int)
        logits, _ = target(torch.cat([block, block]), cond_idx=None, input_pos=input_pos)
        lcfg = cfg_combine(logits, cfg_scale)

        if mode == "stochastic":
            p_probs = torch.softmax(
                topk_filter(lcfg[0] / max(temperature, 1e-5), top_k), dim=-1)  # (L, V)
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
            if corrected is None:  # all drafts accepted -> bonus from p at last position
                corrected = torch.multinomial(p_probs[draft_len - 1], 1, generator=gen).view(())
            seq[n: n + accepted] = block[0, 1: 1 + accepted]
            seq[n + accepted] = corrected
        else:
            preds = target_choice(lcfg, n)  # logits index j predicts seq[n+j]
            if draft_len > 1:
                matches = (block[0, 1:] == preds[0, :-1]).int()
                accepted = int(torch.cumprod(matches, dim=0).sum().item())
            else:
                accepted = 0
            seq[n: n + accepted] = block[0, 1: 1 + accepted]
            seq[n + accepted] = preds[0, accepted]

        new_tokens = accepted + 1
        acceptance_lengths.append(new_tokens)
        drafter.append_context(drafter.fuse(capture.concat())[:, :new_tokens, :], freqs)
        n += new_tokens

    elapsed = _cuda_time() - t0
    return seq, {
        "total_s": elapsed,
        "draft_s": draft_time,
        "verify_s": elapsed - draft_time,
        "steps": len(acceptance_lengths),
        "tau": sum(acceptance_lengths) / len(acceptance_lengths),
        "acceptance_lengths": acceptance_lengths,
    }
