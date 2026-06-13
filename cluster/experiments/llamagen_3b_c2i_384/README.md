# llamagen_3b_c2i_384

LlamaGen-3B class-conditional speculative decoding with a trained DFlash drafter.

## Target

- **Model**: LlamaGen-c2i-3B (3.1B, 32 layers, dim 3200, 32 heads).
- **Resolution**: 384×384 -> 24×24 grid = **576 image tokens** per image.
- **Conditioning**: 1 class-token prefix (1000 ImageNet classes + null).
- **Sampling**: CFG 4.0, temperature 1.0, top-k 2000 (LlamaGen-c2i defaults).

## Why this experiment

- Cluster replica of the 5080 pilot (same model + recipe) -> validates that
  results transfer from consumer to data-center hardware.
- Direct comparison vs LANTERN's 3B-target ablation (the only published spec
  decoding number we have on this exact model).

## Drafter

- 5 layers at target width (dim 3200), ~674M params, bf16.
- Per-layer KV injection of fused 5-layer target features (layers 1, 6, 11, 16, 21).
- 16-token blocks.

## Run

```
./cluster/experiments/llamagen_3b_c2i_384/run.sh
```

Outputs in `$DFLASH_RUNS/llamagen_3b_c2i_384/`.
