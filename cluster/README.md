# DFlash Visual - Cluster Runbook

How to run the DFlash visual drafter pipeline on a SLURM cluster.

## Layout on the cluster

```
$SCRATCH/dflash_visual/          (writable scratch space, target ~50 GB)
  code/                          (this git repo, cloned)
  pretrained/                    (downloaded once, shared across runs)
    llamagen/                    (LlamaGen target weights + VQ tokenizers)
    t5/                          (FLAN-T5-XL text encoder for t2i)
  data/coco/                     (COCO captions for t2i prompts)
  runs/<EXP_NAME>/               (one folder per experiment)
    config.json                  (snapshot of the JSON used)
    data/                        (generated token shards, deletable after train)
    checkpoints/                 (latest.pt + final.pt)
    logs/                        (sbatch stdout/stderr per job)
    results/                     (results_*.md, results_*.json, sample images)
```

Set `$SCRATCH` to your cluster's writable space (e.g. `/scratch300/$USER`).

## One-time setup

Run `cluster/scripts/setup_env.sh` once to:
1. Create conda env at `$SCRATCH/conda_envs/dflash_visual`
2. Install PyTorch (cu128 wheels), bitsandbytes, transformers, etc.
3. Download pretrained models into `$SCRATCH/dflash_visual/pretrained/`

```
sbatch --gres=gpu:1 --time=1:00:00 -o setup.log cluster/scripts/setup_env.sh
```

## Running an experiment

Each experiment is defined by a JSON config under `cluster/configs/`. Pipeline:

```
EXP=llamagen_xl_t2i_stage2

# 1. Generate training data (split into N parallel jobs via SLURM job array)
sbatch --array=0-3 --gres=gpu:1 cluster/scripts/datagen.sh $EXP

# 2. Train the drafter (single job, single GPU; depends on (1))
sbatch --dependency=afterok:$DATAGEN_JOBID --gres=gpu:1 \
       cluster/scripts/train.sh $EXP

# 3. Evaluate (parallel across prompt subsets; depends on (2))
sbatch --array=0-3 --dependency=afterok:$TRAIN_JOBID --gres=gpu:1 \
       cluster/scripts/eval.sh $EXP
```

See `cluster/scripts/pipeline.sh` for an end-to-end submitter that chains
these with proper dependencies.

## Adding a new experiment

1. Drop a new `cluster/configs/<EXP_NAME>.json` (start from an existing one).
2. Submit the three sbatch jobs above with `EXP=<EXP_NAME>`.
3. Each job writes only into `$SCRATCH/dflash_visual/runs/$EXP_NAME/`, so
   experiments never collide.
