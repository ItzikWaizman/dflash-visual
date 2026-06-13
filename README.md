# Visual DFlash

DFlash-style block-diffusion speculative decoding for visual autoregressive
models (LlamaGen, Lumina-mGPT, Anole, Emu3, ...).

## What this repo contains

- `dflash_visual_drafter.py` - the 5-layer block-diffusion drafter and the
  parallel verification engine. Shared by every experiment.
- `dflash_visual_poc.py` - the original mock-drafter POC (lossless engine
  validation, eps-sweep speedup curves).
- `generate_training_data.py` - c2i datagen via self-distillation from a
  LlamaGen-c2i target.
- `generate_training_data_t2i.py` - t2i datagen; uses FLAN-T5-XL to encode
  prompts, samples LlamaGen-T2I targets, caches T5 features once per run.
- `train_drafter.py` / `train_drafter_t2i.py` - drafter training, task-specific.
- `eval_real_drafter.py` / `eval_real_drafter_t2i.py` - speculative decoding
  vs sequential baseline across greedy / Gumbel / Leviathan-stochastic.
- `LlamaGen/` - vendored target-model code.
- `cluster/` - SLURM runbook, env.sh, JSON experiment configs, sbatch scripts.

## Local pilot (RTX 5080) - in progress

Class-conditional LlamaGen-3B at 384x384, 60K self-distilled sequences,
4-epoch drafter training. Validates the recipe; the τ number is the
calibration baseline. Pipeline:

```
python generate_training_data.py --per-class 60                # ~30h, runs once
python train_drafter.py --data data/train_tokens               # ~20h
python eval_real_drafter.py --ckpt checkpoints/latest.pt       # ~30min
```

## Cluster runs (paper experiments)

See `cluster/README.md` for full instructions. TL;DR for a new experiment:

```
EXP=llamagen_xl_t2i_stage2
./cluster/scripts/pipeline.sh $EXP   # submits datagen -> train -> eval
                                     # with --dependency=afterok chains
```

The standard model set we benchmark on (each = one JSON config):

| Experiment                  | Target            | Task | Compared against |
| --------------------------- | ----------------- | ---- | ---------------- |
| llamagen_3b_c2i_384         | LlamaGen-3B       | c2i  | LANTERN (3B ablation) |
| llamagen_xl_t2i_stage2      | LlamaGen-XL S2    | t2i  | LANTERN, LANTERN++, SJD++ |
| lumina_mgpt_7b_768          | Lumina-mGPT-7B    | t2i  | SJD, SJD-PAC, LANTERN++, SJD++ |
| anole_7b_512                | Anole-7B          | t2i  | LANTERN++, SJD |
| emu3_8b                     | Emu3-8B           | t2i  | SJD++ |
