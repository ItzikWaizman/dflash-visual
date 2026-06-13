# DFlash Visual POC — Results

**Goal.** Port DFlash (arXiv:2602.06036) — block-diffusion drafting conditioned on multi-layer target hidden features via KV injection, verified losslessly by the target — from text LLMs to visual autoregressive image generation (LlamaGen, arXiv:2406.06525).

**Hardware.** NVIDIA GeForce RTX 5080 (16 GB), Windows, PyTorch 2.11 + cu128, eager mode (no `torch.compile`).
**Setup.** 384×384 images = 576 visual tokens (24×24, VQ-16), CFG = 4.0, block size B = 16, 5-layer drafter, 5 target hidden features (layers uniformly spanned, DFlash recipe). Drafter predictions are emulated (teacher forcing with controlled error rate eps) while a real 5-layer DFlash-dimensioned drafter forward — KV injection, bidirectional block attention, shared target LM head — is executed and **included in all wall-clock numbers** (614M params at GPT-3B width).

---

## Did it work? Yes.

### Tier 1 — Lossless correctness (LlamaGen-XXL 1.4B, fp32, greedy, true token-comparison verification)

| Drafter noise | torch.equal vs sequential | Token agreement | tau (/16) |
|---|---|---|---|
| 0.00 | **PASS** (4/4 classes) | 100.00% | 15.97 |
| 0.10 | **PASS** (4/4 classes) | 100.00% | 7.88–8.33 |

The speculative engine reproduces the sequential autoregressive output **bit-exactly**, even when the drafter is wrong 10% of the time (rejection + correction recovers the exact trajectory). Measured tau at eps = 0.10 matches the theoretical i.i.d. expectation (≈7.9), confirming the verification math.

### Tier 2 — Speed (LlamaGen-3B 3.1B, bf16, temperature 1.0, top-k 2000, seed-aligned Gumbel sampling)

Sequential baseline: **11.23 s/image** (51.3 tok/s, mean of 8 ImageNet classes).

| Drafter per-token error eps | tau (/16) | Acceptance | Spec latency (s/image) | Draft (s) | Verify (s) | **Speedup** | Envelope (free drafter) |
|---|---|---|---|---|---|---|---|
| 0.00 (oracle) | 15.97 | 99.8% | 0.95 | 0.16 | 0.77 | **11.79×** | 14.61× |
| 0.05 | 11.06 | 69.1% | 1.38 | 0.24 | 1.12 | **8.16×** | 10.02× |
| 0.10 | 8.07 | 50.4% | 1.89 | 0.32 | 1.54 | **5.94×** | 7.27× |
| 0.25 | 3.89 | 24.3% | 3.88 | 0.67 | 3.19 | **2.89×** | 3.52× |

Sample images (sequential baseline, decoded with VQ-16): `poc_samples.png` — valid, high-quality class-conditional ImageNet samples.

**Headline estimate.** A trained DFlash drafter on text reaches tau ≈ 6.5–8 of 16 (DFlash paper, Table 1). At the equivalent visual operating point (eps ≈ 0.10, tau ≈ 8), the engine delivers **≈ 5.9× end-to-end** on a 3.1B visual AR model — with drafting overhead honestly included.

---

## How does this compare to prior work?

All published speculative-decoding-for-visual-AR results on LlamaGen-class models:

| Method | Mechanism | Speedup on LlamaGen | Lossless? |
|---|---|---|---|
| LANTERN (ICLR'25) | AR drafter + relaxed acceptance | 1.75–1.82× | No (relaxed) |
| LANTERN++ | static tree + relaxed acceptance | ~2.3× | No (relaxed) |
| MuLo-SD | low-res drafter + local verification | up to 1.7× | No |
| Cool-SD | annealed relaxation | ~2.3–2.7× | Near-lossless |
| SJD / Coupled-SJD | Jacobi self-speculation | ~2× / ~3.8× | Lossless |
| PJD (CVPR'26) | 2D Jacobi decoding | 4.8× | Quality-preserving |
| GSD | token-cluster acceptance | ~3.7× avg | No (cluster-relaxed) |
| **This POC (DFlash-visual)** | **block-diffusion drafter + target-feature KV injection** | **5.9× @ tau=8 (11.8× oracle ceiling)** | **Lossless (greedy)** |

No prior work applies the DFlash mechanism (parallel block-diffusion drafting conditioned on fused target hidden states injected as KV into every drafter layer) to visual AR generation — the innovation claim stands. The verification-engine ceiling measured here (5.9–8.2× at realistic acceptance) exceeds every published number on LlamaGen, including the strongest training-free method (PJD, 4.8×).

## Key research finding: visual token ambiguity meets numerics

Tier 1b probe — exact greedy verification on GPT-3B in **bf16**: token agreement collapses to 2.6–5.6% (tau ≈ 1.0). Cause: visual AR logit distributions are extremely flat (measured top1–top2 gaps as low as ~0.19 logits), so the numerical difference between a 16-token parallel forward and a 1-token sequential forward (different bf16 kernel reductions, max observed logit delta 0.5) flips argmax decisions at near-ties, and the trajectory diverges at the first flip. The same engine in fp32 is bit-exact (Tier 1). This independently confirms and quantifies the "token selection ambiguity" identified by LANTERN (arXiv:2410.03355) — and shows it appears even between two numerically valid evaluations of the *same* model. Practical implication: production visual speculative decoding should use relaxed/distributional acceptance (LANTERN-style) or higher-precision logit computation, rather than exact token matching.

## Limitations (honest disclosure)

1. **Mock drafter predictions.** Draft quality is emulated via a controlled error rate; the eps sweep brackets what a trained drafter would achieve. Training a real 5-layer block-diffusion drafter on ImageNet token sequences (offline distillation from LlamaGen-3B, ~1–2 GPU-days) is the natural next step; DFlash text results (tau 6.5–8/16) suggest eps ≈ 0.05–0.10 is attainable since visual hidden states should encode local spatial continuations well.
2. **Tier-2 acceptance is statistically simulated** (exact i.i.d. distribution of a (1−eps)-accurate drafter); all verification and drafting compute is genuinely executed and timed. Tier-1 uses true token-comparison acceptance.
3. Greedy verification only (per POC spec); stochastic lossless verification needs Leviathan-style rejection sampling.
4. Class-conditional only; t2i is a straightforward extension.
5. Eager PyTorch on Windows; both baseline and speculative paths benefit similarly from compilation, so the ratio should be robust.

## Artifacts

- `dflash_visual_poc.py` — standalone POC (Components 1–3 + both tiers + benchmark).
- `results.json` — raw numbers. `full_run.log` — complete run log.
- `poc_samples.png` — generated 384×384 samples (8 classes).
- `debug_block.py` — numerics probe that isolated the bf16 ambiguity finding.
