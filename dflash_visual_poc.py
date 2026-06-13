"""
DFlash Visual POC: Block Diffusion Speculative Decoding for LlamaGen
=====================================================================

Ports the DFlash (arXiv:2602.06036) speculative decoding architecture from
text LLMs to visual autoregressive image generation (LlamaGen, arXiv:2406.06525).

Components, per POC spec:
  1. HiddenStateExtractor  - forward hooks on 5 uniformly spaced target layers,
                             concat + linear projection + RMSNorm (DFlash "fused
                             target context feature"), KV-injected into the drafter.
  2. MockBlockDrafter      - drafts a block of B=16 visual tokens in one shot.
                             Prediction quality is emulated (teacher forcing with a
                             controlled error rate), while the compute cost of a real
                             5-layer DFlash drafter is faithfully executed and timed.
  3. Parallel verification - single batched forward of the drafted block through
                             the target (with CFG), greedy accept-until-mismatch,
                             bonus-token correction, KV-cache window shift.

Two evaluation tiers (motivated by visual "token selection ambiguity", cf.
LANTERN arXiv:2410.03355): visual AR logit distributions are extremely flat, so
in bf16 the kernel-shape difference between a 16-token parallel forward and a
1-token sequential forward flips argmax decisions at near-ties. Therefore:

  Tier 1 (correctness): LlamaGen-XXL in float32 (TF32 disabled), greedy decoding,
        teacher-forced drafts, REAL token-comparison verification. Asserts
        torch.equal(speculative, sequential) -> proves the engine is lossless
        when numerics are stable. Also measures bf16 token agreement on the
        large model as a quantified ambiguity finding.

  Tier 2 (speed): LlamaGen-3B in bfloat16 with realistic stochastic sampling
        (temperature 1.0, top-k 2000, position-indexed Gumbel noise so baseline
        and speculative decoding are seed-aligned). Drafter accuracy is simulated
        statistically: each drafted position is independently correct with
        probability (1 - eps), giving the exact acceptance-length distribution of
        a drafter with per-token accuracy (1 - eps). All verification compute
        (block forward, argmax, comparison) plus the full drafter-sim compute is
        executed and included in wall-clock time.

Run:  python dflash_visual_poc.py            (both tiers)
      python dflash_visual_poc.py --tier 1   (correctness only)
      python dflash_visual_poc.py --tier 2   (speed only)
"""
import argparse
import gc
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "LlamaGen"))

from autoregressive.models.gpt import GPT_models, RMSNorm  # noqa: E402
from tokenizer.tokenizer_image.vq_model import VQ_models  # noqa: E402


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def cfg_combine(logits: torch.Tensor, cfg_scale: float) -> torch.Tensor:
    """logits: (2*b, L, V) with cond half first -> (b, L, V)."""
    cond, uncond = torch.split(logits, logits.shape[0] // 2, dim=0)
    return uncond + (cond - uncond) * cfg_scale


def choose_tokens(logits: torch.Tensor, gumbel: torch.Tensor | None,
                  temperature: float, top_k: int) -> torch.Tensor:
    """Greedy if temperature==0, else Gumbel-max sampling with top-k filtering.
    logits: (1, L, V); gumbel: (L, V) position-indexed noise. Returns (1, L)."""
    if temperature == 0.0 or gumbel is None:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k > 0:
        thresh = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = torch.where(logits < thresh, torch.full_like(logits, -float("inf")), logits)
    return (logits + gumbel.unsqueeze(0)).argmax(dim=-1)


def make_gumbel(num_tokens: int, vocab: int, seed: int, device) -> torch.Tensor:
    """Position-indexed Gumbel noise: same tensor for baseline and speculative
    runs => identical sampling decisions wherever logits agree."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    u = torch.rand(num_tokens, vocab, generator=g)
    return (-torch.log(-torch.log(u + 1e-20) + 1e-20)).to(device)


def build_target_layer_ids(num_target_layers: int, num_features: int):
    """Uniformly sample feature layers between layer 1 and n-3 (DFlash recipe)."""
    if num_features == 1:
        return [num_target_layers // 2]
    start, end = 1, num_target_layers - 3
    span = end - start
    return [int(round(start + (i * span) / (num_features - 1))) for i in range(num_features)]


# ---------------------------------------------------------------------------
# Component 1: Hidden State & KV Injection Plumbing
# ---------------------------------------------------------------------------

class HiddenStateExtractor:
    """Captures intermediate hidden states of selected LlamaGen layers via
    forward hooks and fuses them into a single DFlash-style context feature."""

    def __init__(self, target: nn.Module, num_features: int, dtype, device):
        self.layer_ids = build_target_layer_ids(target.n_layer, num_features)
        dim = target.config.dim
        self.fc = nn.Linear(num_features * dim, dim, bias=False).to(device=device, dtype=dtype)
        self.norm = RMSNorm(dim).to(device=device, dtype=dtype)
        self._captured: dict[int, torch.Tensor] = {}
        self._handles = []
        for lid in self.layer_ids:
            self._handles.append(
                target.layers[lid].register_forward_hook(self._make_hook(lid))
            )

    def _make_hook(self, lid: int):
        def hook(_module, _inputs, output):
            self._captured[lid] = output  # (2b, L, D) residual stream after block
        return hook

    @torch.no_grad()
    def fused_feature(self) -> torch.Tensor:
        """Concat captured features (cond half only) and project. (1, L, D)"""
        feats = [self._captured[lid] for lid in self.layer_ids]
        cat = torch.cat(feats, dim=-1)
        cond_half = cat[: cat.shape[0] // 2]
        return self.norm(self.fc(cond_half))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Component 2: Block-Diffusion Drafter (mock prediction, realistic compute)
# ---------------------------------------------------------------------------

class DrafterSimLayer(nn.Module):
    """One transformer layer with DFlash KV injection: queries come from the
    masked block tokens; keys/values are [projected target context ; block]."""

    def __init__(self, dim: int, n_head: int):
        super().__init__()
        self.n_head, self.head_dim = n_head, dim // n_head
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        hidden = int(2 * 4 * dim / 3)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.ln1 = RMSNorm(dim)
        self.ln2 = RMSNorm(dim)

    def project_ctx_kv(self, ctx: torch.Tensor):
        b, L, _ = ctx.shape
        k = self.k_proj(ctx).view(b, L, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(ctx).view(b, L, self.n_head, self.head_dim).transpose(1, 2)
        return k, v

    def forward(self, x: torch.Tensor, ctx_k: torch.Tensor, ctx_v: torch.Tensor):
        b, L, _ = x.shape
        h = self.ln1(x)
        q = self.q_proj(h).view(b, L, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, L, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, L, self.n_head, self.head_dim).transpose(1, 2)
        k = torch.cat([ctx_k, k], dim=2)
        v = torch.cat([ctx_v, v], dim=2)
        attn = F.scaled_dot_product_attention(q, k, v)  # bidirectional in block
        x = x + self.o_proj(attn.transpose(1, 2).reshape(b, L, -1))
        h = self.ln2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class MockBlockDrafter(nn.Module):
    """Drafts B visual tokens in a single parallel forward.

    Compute: a real 5-layer DFlash-dimensioned forward (KV injection of fused
    target features, bidirectional block attention, shared target LM head) is
    executed every draft step so wall-clock numbers include honest drafting
    overhead. Its prediction is replaced by teacher forcing (we benchmark the
    verification engine, not draft-model quality).
    """

    def __init__(self, target: nn.Module, num_layers: int, vocab_size: int, dtype, device):
        super().__init__()
        dim, n_head = target.config.dim, target.config.n_head
        self.vocab_size = vocab_size
        self.layers = nn.ModuleList([DrafterSimLayer(dim, n_head) for _ in range(num_layers)])
        self.norm = RMSNorm(dim)
        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        self.to(device=device, dtype=dtype)
        self.target = [target]  # hide from .parameters()
        self._ctx_k: list = [None] * num_layers
        self._ctx_v: list = [None] * num_layers

    def reset_cache(self):
        self._ctx_k = [None] * len(self.layers)
        self._ctx_v = [None] * len(self.layers)

    @torch.no_grad()
    def append_context(self, fused_feature: torch.Tensor):
        """KV-inject newly accepted target features into every draft layer's cache."""
        for i, layer in enumerate(self.layers):
            k, v = layer.project_ctx_kv(fused_feature)
            if self._ctx_k[i] is None:
                self._ctx_k[i], self._ctx_v[i] = k, v
            else:
                self._ctx_k[i] = torch.cat([self._ctx_k[i], k], dim=2)
                self._ctx_v[i] = torch.cat([self._ctx_v[i], v], dim=2)

    @torch.no_grad()
    def sim_forward(self, anchor_token: torch.Tensor, draft_len: int):
        """Realistic-compute pass; output discarded (mock prediction)."""
        target = self.target[0]
        emb = target.tok_embeddings(anchor_token.view(1, 1))
        x = torch.cat([emb, self.mask_embedding.expand(1, draft_len - 1, -1)], dim=1)
        for i, layer in enumerate(self.layers):
            x = layer(x, self._ctx_k[i], self._ctx_v[i])
        _ = target.output(self.norm(x))  # shared target LM head (timed, unused)

    @torch.no_grad()
    def draft_teacher_forced(self, anchor_token: torch.Tensor, reference_seq: torch.Tensor,
                             start: int, draft_len: int, noise: float,
                             gen: torch.Generator | None):
        """Returns (1, draft_len): [anchor, d_1 .. d_{L-1}] where d_i is the
        reference token, corrupted independently with probability `noise`."""
        self.sim_forward(anchor_token, draft_len)
        block = torch.empty(1, draft_len, dtype=torch.long, device=anchor_token.device)
        block[0, 0] = anchor_token
        if draft_len > 1:
            proposal = reference_seq[start: start + draft_len - 1].clone()
            if noise > 0 and gen is not None:
                flip = torch.rand(proposal.shape, device=proposal.device, generator=gen) < noise
                rand_ids = torch.randint(0, self.vocab_size, proposal.shape,
                                         device=proposal.device, generator=gen)
                proposal = torch.where(flip, rand_ids, proposal)
            block[0, 1:] = proposal
        return block


# ---------------------------------------------------------------------------
# Baseline: sequential autoregressive decoding (LlamaGen default loop)
# ---------------------------------------------------------------------------

@torch.no_grad()
def baseline_generate(model, class_idx: int, num_tokens: int, cfg_scale: float,
                      temperature: float, top_k: int, gumbel, device):
    cond = torch.tensor([class_idx], device=device)
    cond_combined = torch.cat([cond, torch.full_like(cond, model.num_classes)])
    T = 1
    with torch.device(device):
        model.setup_caches(max_batch_size=2, max_seq_length=T + num_tokens,
                           dtype=model.tok_embeddings.weight.dtype)
    t0 = cuda_time()
    input_pos = torch.arange(0, T, device=device)
    logits, _ = model(None, cond_combined, input_pos)
    g0 = gumbel[0:1] if gumbel is not None else None
    next_token = choose_tokens(cfg_combine(logits[:, -1:], cfg_scale), g0, temperature, top_k)

    seq = torch.empty(num_tokens, dtype=torch.long, device=device)
    seq[0] = next_token[0, 0]
    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    for i in range(1, num_tokens):
        x = next_token.view(1, 1)
        logits, _ = model(torch.cat([x, x]), cond_idx=None, input_pos=input_pos)
        gi = gumbel[i:i + 1] if gumbel is not None else None
        next_token = choose_tokens(cfg_combine(logits[:, -1:], cfg_scale), gi, temperature, top_k)
        seq[i] = next_token[0, 0]
        input_pos += 1
    elapsed = cuda_time() - t0
    return seq, elapsed


# ---------------------------------------------------------------------------
# Component 3: Parallel Visual Verification Engine (DFlash-style)
# ---------------------------------------------------------------------------

@torch.no_grad()
def speculative_generate(model, drafter: MockBlockDrafter, extractor: HiddenStateExtractor,
                         class_idx: int, num_tokens: int, cfg_scale: float,
                         block_size: int, noise: float, seed: int, reference_seq,
                         temperature: float, top_k: int, gumbel, device,
                         acceptance_mode: str = "exact"):
    """acceptance_mode:
      'exact'       - accept while drafted token == target choice (true verification;
                      requires numerically stable setting, see Tier 1).
      'statistical' - acceptance length drawn from the exact distribution of a
                      drafter with i.i.d. per-token accuracy (1 - noise); all
                      verification compute is still executed (timing-faithful).
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed * 7919 + class_idx)
    cond = torch.tensor([class_idx], device=device)
    cond_combined = torch.cat([cond, torch.full_like(cond, model.num_classes)])
    T = 1
    with torch.device(device):
        model.setup_caches(max_batch_size=2, max_seq_length=T + num_tokens,
                           dtype=model.tok_embeddings.weight.dtype)
    drafter.reset_cache()

    t0 = cuda_time()
    draft_time = 0.0
    verify_time = 0.0

    # Prefill: class token at position 0 -> first visual token + initial features
    input_pos = torch.arange(0, T, device=device)
    logits, _ = model(None, cond_combined, input_pos)
    g0 = gumbel[0:1] if gumbel is not None else None
    first = choose_tokens(cfg_combine(logits[:, -1:], cfg_scale), g0, temperature, top_k)
    drafter.append_context(extractor.fused_feature())

    seq = torch.empty(num_tokens, dtype=torch.long, device=device)
    seq[0] = first[0, 0]
    n = 1
    acceptance_lengths = []

    while n < num_tokens:
        remaining = num_tokens - n
        draft_len = min(block_size, remaining)

        td = cuda_time()
        noise_for_draft = noise if acceptance_mode == "exact" else 0.0
        block = drafter.draft_teacher_forced(seq[n - 1], reference_seq, n,
                                             draft_len, noise_for_draft, gen)
        draft_time += cuda_time() - td

        tv = cuda_time()
        # one parallel forward: positions n .. n+draft_len-1 (cls token at pos 0)
        input_pos = torch.arange(n, n + draft_len, device=device, dtype=torch.int)
        logits, _ = model(torch.cat([block, block]), cond_idx=None, input_pos=input_pos)
        gblk = gumbel[n - 1: n - 1 + draft_len] if gumbel is not None else None
        preds = choose_tokens(cfg_combine(logits, cfg_scale), gblk, temperature, top_k)

        # token-comparison verification (always computed; used in 'exact' mode)
        if draft_len > 1:
            matches = (block[0, 1:] == preds[0, :-1]).int()
            accepted_exact = int(torch.cumprod(matches, dim=0).sum().item())
        else:
            accepted_exact = 0

        if acceptance_mode == "exact":
            accepted = accepted_exact
        else:
            ok = (torch.rand(draft_len - 1, device=device, generator=gen) >= noise).int() \
                if draft_len > 1 else torch.zeros(0, device=device, dtype=torch.int)
            accepted = int(torch.cumprod(ok, dim=0).sum().item()) if draft_len > 1 else 0

        # accepted drafted tokens + 1 bonus/correction token from the target
        seq[n: n + accepted] = block[0, 1: 1 + accepted]
        seq[n + accepted] = preds[0, accepted]
        new_tokens = accepted + 1
        acceptance_lengths.append(new_tokens)

        # features of accepted positions feed the next draft (DFlash semantics)
        drafter.append_context(extractor.fused_feature()[:, :new_tokens, :])
        n += new_tokens
        verify_time += cuda_time() - tv

    elapsed = cuda_time() - t0
    stats = {
        "total_s": elapsed,
        "draft_s": draft_time,
        "verify_s": verify_time,
        "steps": len(acceptance_lengths),
        "tau": sum(acceptance_lengths) / len(acceptance_lengths),
    }
    return seq, stats


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_gpt(name: str, ckpt_path: str, dtype, device, codebook_size, latent_size):
    print(f"Building {name} ({dtype}) ...", flush=True)
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    model = GPT_models[name](vocab_size=codebook_size, block_size=latent_size ** 2,
                             num_classes=1000, cls_token_num=1, model_type="c2i")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    weights = ckpt if "model" not in ckpt else ckpt["model"]
    missing, unexpected = model.load_state_dict(weights, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    del ckpt, weights
    gc.collect()
    model.to(device=device, dtype=dtype).eval()
    gc.collect()
    torch.cuda.empty_cache()
    n = sum(p.numel() for p in model.parameters())
    print(f"  {name}: {n/1e9:.2f}B params | GPU mem {torch.cuda.memory_allocated()/1e9:.1f} GB",
          flush=True)
    return model


def free_model(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------

def run_tier1(args, device):
    """Correctness: exact verification + torch.equal, fp32, greedy."""
    print("\n" + "=" * 70)
    print("TIER 1 - LOSSLESS CORRECTNESS (LlamaGen-XXL, float32, greedy)")
    print("=" * 70, flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    latent = args.image_size // args.downsample_size
    num_tokens = latent ** 2
    model = load_gpt("GPT-XXL", args.xxl_ckpt, torch.float32, device,
                     args.codebook_size, latent)
    extractor = HiddenStateExtractor(model, args.num_features, torch.float32, device)
    drafter = MockBlockDrafter(model, args.drafter_layers, args.codebook_size,
                               torch.float32, device)
    drafter.eval()

    # warmup
    wseq, _ = baseline_generate(model, args.classes[0], num_tokens, args.cfg_scale,
                                0.0, 0, None, device)
    speculative_generate(model, drafter, extractor, args.classes[0], num_tokens,
                         args.cfg_scale, args.block_size, 0.0, 0, wseq,
                         0.0, 0, None, device, "exact")

    rows = []
    classes = args.classes[: args.tier1_classes]
    for noise in [0.0, 0.10]:
        runs, losses = [], []
        for cls in classes:
            base_seq, base_t = baseline_generate(model, cls, num_tokens, args.cfg_scale,
                                                 0.0, 0, None, device)
            seq, st = speculative_generate(model, drafter, extractor, cls, num_tokens,
                                           args.cfg_scale, args.block_size, noise, 0,
                                           base_seq, 0.0, 0, None, device, "exact")
            eq = torch.equal(seq, base_seq)
            agree = (seq == base_seq).float().mean().item()
            losses.append(eq)
            runs.append({"cls": cls, "tau": st["tau"], "equal": eq, "agree": agree,
                         "base_t": base_t, "spec_t": st["total_s"]})
            print(f"  noise={noise:.2f} class {cls:4d}: torch.equal={eq} "
                  f"agreement={100*agree:.2f}% tau={st['tau']:.2f}", flush=True)
        rows.append({
            "noise": noise,
            "all_equal": all(losses),
            "tau": sum(r["tau"] for r in runs) / len(runs),
            "agree": sum(r["agree"] for r in runs) / len(runs),
        })

    extractor.remove()
    free_model(model, drafter, extractor)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return rows, num_tokens


def run_tier1b_bf16_ambiguity(args, model, drafter, extractor, num_tokens, device):
    """Quantify visual token ambiguity: exact verification on bf16 large model."""
    print("\n--- Tier 1b: bf16 token-ambiguity probe on GPT-3B (exact mode, greedy) ---",
          flush=True)
    rows = []
    for cls in args.classes[:2]:
        base_seq, _ = baseline_generate(model, cls, num_tokens, args.cfg_scale,
                                        0.0, 0, None, device)
        seq, st = speculative_generate(model, drafter, extractor, cls, num_tokens,
                                       args.cfg_scale, args.block_size, 0.0, 0,
                                       base_seq, 0.0, 0, None, device, "exact")
        agree = (seq == base_seq).float().mean().item()
        rows.append({"cls": cls, "tau": st["tau"], "agree": agree})
        print(f"  class {cls:4d}: agreement={100*agree:.2f}% tau={st['tau']:.2f}/16",
              flush=True)
    return rows


def run_tier2(args, device):
    """Speed: bf16 GPT-3B, stochastic sampling, statistical acceptance."""
    print("\n" + "=" * 70)
    print("TIER 2 - SPEED BENCHMARK (LlamaGen-3B, bfloat16, temp=1.0 top-k=2000)")
    print("=" * 70, flush=True)
    dtype = torch.bfloat16
    latent = args.image_size // args.downsample_size
    num_tokens = latent ** 2

    model = load_gpt("GPT-3B", args.gpt_ckpt, dtype, device, args.codebook_size, latent)
    extractor = HiddenStateExtractor(model, args.num_features, dtype, device)
    drafter = MockBlockDrafter(model, args.drafter_layers, args.codebook_size, dtype, device)
    drafter.eval()
    print(f"Drafter sim: {sum(p.numel() for p in drafter.parameters())/1e6:.0f}M params, "
          f"feature layers {extractor.layer_ids}", flush=True)

    # warmup
    g = make_gumbel(num_tokens, args.codebook_size, 1234, device)
    wseq, _ = baseline_generate(model, args.classes[0], num_tokens, args.cfg_scale,
                                args.temperature, args.top_k, g, device)
    speculative_generate(model, drafter, extractor, args.classes[0], num_tokens,
                         args.cfg_scale, args.block_size, 0.0, 0, wseq,
                         args.temperature, args.top_k, g, device, "statistical")
    torch.cuda.empty_cache()

    ambiguity = run_tier1b_bf16_ambiguity(args, model, drafter, extractor, num_tokens, device)

    print("\n--- Baseline: sequential sampling ---", flush=True)
    baselines, gumbels, base_times = {}, {}, []
    for cls in args.classes:
        gum = make_gumbel(num_tokens, args.codebook_size, 1000 + cls, device)
        seq, t = baseline_generate(model, cls, num_tokens, args.cfg_scale,
                                   args.temperature, args.top_k, gum, device)
        baselines[cls], gumbels[cls] = seq, gum
        base_times.append(t)
        print(f"  class {cls:4d}: {t:.2f}s ({num_tokens/t:.1f} tok/s)", flush=True)
    base_mean = sum(base_times) / len(base_times)

    results = []
    for noise in args.noise_levels:
        print(f"\n--- DFlash visual speculative decoding (drafter error eps={noise:.2f}) ---",
              flush=True)
        runs = []
        for cls in args.classes:
            for seed in (args.seeds if noise > 0 else args.seeds[:1]):
                seq, st = speculative_generate(
                    model, drafter, extractor, cls, num_tokens, args.cfg_scale,
                    args.block_size, noise, seed, baselines[cls],
                    args.temperature, args.top_k, gumbels[cls], device, "statistical")
                runs.append(st)
        mean = lambda k: sum(r[k] for r in runs) / len(runs)
        row = {
            "noise": noise,
            "spec_s": mean("total_s"),
            "draft_s": mean("draft_s"),
            "verify_s": mean("verify_s"),
            "tau": mean("tau"),
            "steps": mean("steps"),
            "speedup_real": base_mean / mean("total_s"),
            "speedup_envelope": base_mean / mean("verify_s"),
        }
        results.append(row)
        print(f"  latency {row['spec_s']:.2f}s | draft {row['draft_s']:.2f}s | "
              f"verify {row['verify_s']:.2f}s | tau {row['tau']:.2f}/{args.block_size} | "
              f"speedup {row['speedup_real']:.2f}x (envelope {row['speedup_envelope']:.2f}x)",
              flush=True)

    # decode showcase images from the (valid) sequential baseline samples
    print("\nDecoding sample images ...", flush=True)
    vq_model = VQ_models[args.vq_model](codebook_size=args.codebook_size,
                                        codebook_embed_dim=args.codebook_embed_dim)
    ckpt = torch.load(args.vq_ckpt, map_location="cpu", weights_only=True)
    vq_model.load_state_dict(ckpt["model"])
    del ckpt
    vq_model.to(device=device, dtype=torch.float32).eval()
    from torchvision.utils import save_image
    idx = torch.stack([baselines[c] for c in args.classes])
    qz = [len(args.classes), args.codebook_embed_dim, latent, latent]
    with torch.no_grad():
        samples = vq_model.decode_code(idx.to(torch.int), qz)
    save_image(samples, os.path.join(ROOT, "poc_samples.png"), nrow=4,
               normalize=True, value_range=(-1, 1))
    print("saved poc_samples.png", flush=True)

    extractor.remove()
    free_model(model, drafter, extractor, vq_model)
    return {"baseline_s": base_mean, "base_times": base_times,
            "rows": results, "ambiguity": ambiguity, "num_tokens": num_tokens}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", type=str, default="all", choices=["1", "2", "all"])
    p.add_argument("--gpt-ckpt", default=os.path.join(ROOT, "LlamaGen", "pretrained_models", "c2i_3B_384.pt"))
    p.add_argument("--xxl-ckpt", default=os.path.join(ROOT, "LlamaGen", "pretrained_models", "c2i_XXL_384.pt"))
    p.add_argument("--vq-model", default="VQ-16")
    p.add_argument("--vq-ckpt", default=os.path.join(ROOT, "LlamaGen", "pretrained_models", "vq_ds16_c2i.pt"))
    p.add_argument("--codebook-size", type=int, default=16384)
    p.add_argument("--codebook-embed-dim", type=int, default=8)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--downsample-size", type=int, default=16)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=2000)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--drafter-layers", type=int, default=5)
    p.add_argument("--num-features", type=int, default=5)
    p.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.25])
    p.add_argument("--classes", type=int, nargs="+", default=[207, 360, 387, 974, 88, 979, 417, 279])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--tier1-classes", type=int, default=4)
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    out = {"gpu": torch.cuda.get_device_name(0)}
    if args.tier in ("1", "all"):
        tier1_rows, _ = run_tier1(args, device)
        out["tier1"] = tier1_rows
    if args.tier in ("2", "all"):
        out["tier2"] = run_tier2(args, device)

    # ------------------------------------------------------------------ report
    lines = ["# DFlash Visual POC Results", ""]
    lines.append(f"GPU: {out['gpu']} | image 384x384 = 576 visual tokens | "
                 f"CFG={args.cfg_scale} | block size B={args.block_size} | "
                 f"drafter {args.drafter_layers} layers, {args.num_features} target features")
    lines.append("")
    if "tier1" in out:
        lines.append("## Tier 1 - Lossless correctness (LlamaGen-XXL 1.4B, fp32, greedy, exact verification)")
        lines.append("")
        lines.append("| Drafter noise | torch.equal | Token agreement | tau (/16) |")
        lines.append("|---|---|---|---|")
        for r in out["tier1"]:
            lines.append(f"| {r['noise']:.2f} | {'PASS' if r['all_equal'] else 'FAIL'} | "
                         f"{100*r['agree']:.2f}% | {r['tau']:.2f} |")
        lines.append("")
    if "tier2" in out:
        t2 = out["tier2"]
        lines.append("## Tier 1b - bf16 token-ambiguity probe (GPT-3B, greedy, exact verification)")
        lines.append("")
        for r in t2["ambiguity"]:
            lines.append(f"- class {r['cls']}: token agreement {100*r['agree']:.2f}%, "
                         f"tau {r['tau']:.2f}/16 (parallel-vs-sequential bf16 argmax flips; "
                         f"cf. LANTERN token-selection ambiguity)")
        lines.append("")
        lines.append("## Tier 2 - Speed benchmark (LlamaGen-3B 3.1B, bf16, temp=1.0, top-k=2000)")
        lines.append("")
        lines.append(f"Sequential baseline: **{t2['baseline_s']:.2f} s/image** "
                     f"({t2['num_tokens']/t2['baseline_s']:.1f} tok/s, mean of {len(args.classes)} classes)")
        lines.append("")
        lines.append("| Drafter error eps | tau (/16) | Acceptance | Spec latency (s) | Draft (s) | Verify (s) | Speedup | Envelope speedup |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in t2["rows"]:
            lines.append(f"| {r['noise']:.2f} | {r['tau']:.2f} | {100*r['tau']/args.block_size:.1f}% | "
                         f"{r['spec_s']:.2f} | {r['draft_s']:.2f} | {r['verify_s']:.2f} | "
                         f"**{r['speedup_real']:.2f}x** | {r['speedup_envelope']:.2f}x |")
        lines.append("")
    report = "\n".join(lines)
    print("\n" + report, flush=True)
    with open(os.path.join(ROOT, "results.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(ROOT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results.md / results.json")


if __name__ == "__main__":
    main()
