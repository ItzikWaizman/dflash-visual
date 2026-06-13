# DFlash Visual - Cluster Runbook

How to run the DFlash visual drafter pipeline on a SLURM cluster.

## Layout

```
cluster/
  env.sh                        DFlash-project paths only (sourced AFTER user's env.sh)
  requirements.txt              pip deps installed into existing conda env
  lib/                          reusable, parameterized sbatch entries
    setup_env.sh                one-time: install deps, download LlamaGen weights
    fetch_coco_captions.sh      one-time: download COCO captions for t2i prompts
    datagen.sh                  generic datagen, takes <exp> arg
    train.sh                    generic training, takes <exp> arg
    eval.sh                     generic eval,     takes <exp> arg
  experiments/                  one self-contained folder per experiment
    llamagen_xl_t2i_stage2/
      config.json               experiment config (target, sampling, training)
      run.sh                    submit full datagen->train->eval pipeline
      README.md                 experiment-specific notes + comparison table
    llamagen_3b_c2i_384/
      config.json
      run.sh
      README.md
    ...                         add new experiments here
```

Scratch tree (`$DFLASH_ROOT` = `/scratch300/$USER/dflash_vlm` by default):

```
$DFLASH_ROOT/
  dflash-visual/                git clone of this repo (= $DFLASH_CODE)
  pretrained/                   LlamaGen + T5 weights (shared across experiments)
  data/coco/                    COCO captions JSONL
  runs/<exp>/                   per-experiment outputs
    config.json                 snapshot of the experiment json used
    data/                       generated token shards (deletable after train)
    checkpoints/                latest.pt + final.pt
    logs/                       sbatch + python logs
    results/                    per-task json -> merged results.md / results.json
```

## Convention: every .sh starts with the same 3 lines

```bash
source /scratch300/$USER/env.sh
module load anaconda
conda activate /scratch300/$USER/conda_envs/unlearning
```

then `source "$HERE/../../env.sh"` (or `"$(dirname "$0")/../env.sh"` for scripts
in `lib/`) which sets `DFLASH_ROOT`, `DFLASH_CODE`, `DFLASH_PRETRAINED`,
`DFLASH_DATA`, `DFLASH_RUNS`, `HF_HOME`.

## One-time setup

```bash
mkdir -p /scratch300/$USER/dflash_vlm
cd /scratch300/$USER/dflash_vlm
git clone https://github.com/ItzikWaizman/dflash-visual.git

# install our deps into the existing `unlearning` conda env + download LlamaGen weights
sbatch -A gpu-tad-wolf_v2 -p gpu-tad-pool --qos=owner \
       --gres=gpu:1 --time=2:00:00 --cpus-per-task=2 --mem=16G \
       --chdir /scratch300/$USER/dflash_vlm/dflash-visual \
       -o /scratch300/$USER/dflash_vlm/setup_env.log \
       cluster/lib/setup_env.sh

# COCO captions for t2i prompts
sbatch -A gpu-tad-wolf_v2 -p gpu-tad-pool --qos=owner \
       --gres=gpu:1 --cpus-per-task=2 --time=1:00:00 --mem=8G \
       --chdir /scratch300/$USER/dflash_vlm/dflash-visual \
       -o /scratch300/$USER/dflash_vlm/fetch_coco.log \
       cluster/lib/fetch_coco_captions.sh
```

## Running an experiment

Just call its `run.sh` from the login node:

```bash
./cluster/experiments/llamagen_xl_t2i_stage2/run.sh
```

This submits datagen -> train -> eval with proper `--dependency=afterok` chaining.
You can override any sbatch parameter via env vars, e.g. `DG_ARRAY=0-7 ACCT=... ./run.sh`.

## Adding a new experiment

1. `cp -r cluster/experiments/llamagen_xl_t2i_stage2 cluster/experiments/<NEW_EXP>`
2. Edit `cluster/experiments/<NEW_EXP>/config.json` (set `experiment_name`, target,
   sampling, training, eval).
3. Edit `cluster/experiments/<NEW_EXP>/run.sh` (change the `EXP=` line, tune time
   limits / array sizes for the target's compute profile).
4. Update `cluster/experiments/<NEW_EXP>/README.md` with the comparison table.
5. Run `./cluster/experiments/<NEW_EXP>/run.sh` from the login node.

Each experiment is fully self-contained; experiments never collide because
outputs always live under `$DFLASH_RUNS/<exp>/`.
