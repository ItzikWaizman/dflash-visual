# llamagen_xl_t2i_stage2

LlamaGen-XL Stage II text-to-image speculative decoding with a trained DFlash drafter.

## Target

- **Model**: LlamaGen-T2I-XL Stage II (775M, 32 layers, dim 1536, 24 heads).
- **Resolution**: 512×512 -> 32×32 grid = **1024 image tokens** per image.
- **Text encoder**: FLAN-T5-XL (3B), 120 text tokens as soft-prompt prefix.
- **Sampling**: CFG 7.5, temperature 1.0, top-k 1000 (matches LANTERN/LANTERN++/SJD++).

## Data

- **Prompts**: COCO 2017 train captions (`captions_train2017.jsonl`), 60K samples.
- **Eval prompts**: COCO 2017 val captions, 32 prompts.
- **Datagen**: self-distillation from the target at the inference settings above.

## Drafter

- 5 layers at target width (dim 1536), ~190M params, bf16.
- Per-layer KV injection of fused 5-layer target features (layers 1, 8, 15, 23, 28).
- 16-token blocks, single-step denoising.

## Headline numbers to compare against

| Method      | Reported | Notes |
|-------------|---------:|-------|
| LANTERN     | 2.26x    | greedy decoding |
| LANTERN++   | 3.63x    | EAGLE-1 base + tree drafting + relaxed accept |
| SJD++       | 2.32x    | training-free Jacobi (different model size) |
| **Ours**    | TBD      | trained DFlash drafter, greedy / Gumbel / stochastic modes |

## Run

```
./cluster/experiments/llamagen_xl_t2i_stage2/run.sh
```

Submits datagen (4-way array, 20h wall) -> train (24h) -> eval (2-way array, 4h),
chained via `--dependency=afterok`.

Outputs land in `$DFLASH_RUNS/llamagen_xl_t2i_stage2/`:
```
config.json           snapshot of the experiment json
data/                 generated token shards + T5 feature cache
checkpoints/          latest.pt, final.pt
logs/                 datagen, train, eval logs
results/              per-task json -> merged results.md / results.json
```
