# Cluster workflow contract

How **I** prepare cluster work for **you** (Itzik), and what gets committed to
the repo so you can pull + submit a single command.

---

## 1. Workflow per experiment / phase

1. **You ask** for a phase: e.g. *"give me the sbatch command for datagen on
   `llamagen_xl_t2i_stage2`"*.
2. **I prepare and push** (always before handing back a command):
   - the wrapper `.sh` under `cluster/lib/` (generic) or
     `cluster/experiments/<exp>/run.sh` (experiment-specific)
   - the experiment `config.json` under `cluster/experiments/<exp>/`
     (if applicable)
   - any Python script the `.sh` calls
   - `git add -A && git commit && git push` to `main`
3. **I hand you exactly one line** to copy-paste. Single line, no backslash
   continuations.
4. **You run**:
   ```
   cd /scratch300/$USER/dflash_vlm/dflash-visual
   git pull
   mkdir -p output_logs
   <the one-line sbatch I gave you>
   ```

---

## 2. The one-line `sbatch` format

Every sbatch command I give you obeys this template (single line). **The
executable script is always the last token** (no positional args after it):

```
sbatch -A gpu-tad-wolf_v2 -p gpu-tad-pool --qos=owner --gres=gpu:1 --time=<HH:MM:SS> --cpus-per-task=<N> --mem=<XG> -o output_logs/<jobname>.out --job-name=<jobname> --chdir /scratch300/$USER/dflash_vlm/dflash-visual/ ./cluster/experiments/<exp>/<phase>.sh
```

Per-experiment thin wrappers (`./cluster/experiments/<exp>/{datagen,train,eval}.sh`)
each just `exec ./cluster/lib/<phase>.sh <exp>`, so the lib wrappers stay generic
while the sbatch line ends cleanly with no args.

**Hard rules** (every time, no exceptions):

| Flag | Value | Why |
|---|---|---|
| `-A gpu-tad-wolf_v2` | fixed | your account |
| `-p gpu-tad-pool`    | fixed | your partition |
| `--qos=owner`        | fixed | your QOS |
| `--gres=gpu:1`       | **always present** | per your standing rule; even download/utility jobs |
| `--time=HH:MM:SS`    | sized to phase | datagen ~24h, train ~24-48h, eval ~4h, download ~2h |
| `--cpus-per-task=N`  | sized to phase | 2 for I/O-bound, 4 for training |
| `--mem=XG`           | sized to phase | 8G for download, 32G for datagen/eval, 64G for train |
| `-o output_logs/<jobname>.out` | **always present** | primary debug surface; first place we both look on failure |
| `--job-name=<jobname>` | snake_case, matches log filename | makes `squeue` readable |
| `--chdir /scratch300/$USER/dflash_vlm/dflash-visual/` | fixed | repo root; all wrappers assume cwd = repo root |

For **job arrays** (parallel datagen/eval shards), I add `--array=0-N` and
use `%A_%a` in the log path (`output_logs/<jobname>_%A_%a.out`).

For **chained jobs**, I add `--dependency=afterok:<prev_jobid>` so you can
fire-and-forget the next phase after the previous succeeds.

---

## 3. The `.sh` wrapper format

Every `.sh` script we submit to sbatch follows this exact skeleton.
This is the contract — if a wrapper deviates from it, treat it as a bug.

```bash
#!/bin/bash
# <one-line description>
#
# Usage (one-line sbatch):
#   sbatch -A gpu-tad-wolf_v2 -p gpu-tad-pool --qos=owner --gres=gpu:1 --time=HH:MM:SS --cpus-per-task=N --mem=XG -o output_logs/<jobname>.out --job-name=<jobname> --chdir /scratch300/$USER/dflash_vlm/dflash-visual/ ./cluster/lib/<wrapper>.sh <args>
set -euo pipefail

source /scratch300/$USER/env.sh
module load anaconda
conda activate /scratch300/$USER/conda_envs/unlearning

test -f cluster/env.sh || { echo "ERROR: --chdir must point at repo root" >&2; exit 1; }
source ./cluster/env.sh

# ... do work (almost always: invoke a python script) ...
python <script>.py --config <path> --run-dir "$DFLASH_RUNS/<exp>" --pretrained "$DFLASH_PRETRAINED"
```

**Why each line:**
- `set -euo pipefail` — fail fast, surface real errors in `-o` log.
- `source /scratch300/$USER/env.sh` — your cluster bootstrap (CUDA, modules,
  `$SCRATCH`, etc.). **Mandatory first.**
- `module load anaconda` + `conda activate /scratch300/$USER/conda_envs/unlearning`
  — your standing requirement; every wrapper does this.
- `test -f cluster/env.sh` guard — explicit error if `--chdir` was forgotten,
  instead of cryptic "command not found".
- `source ./cluster/env.sh` — sets `$DFLASH_ROOT`, `$DFLASH_CODE`,
  `$DFLASH_DATA`, `$DFLASH_RUNS`, `$DFLASH_PRETRAINED`, `$HF_HOME`.
- The actual work is **almost always a single `python <script>.py ...`** call.
  Bash glue stays minimal. If logic gets complex, it lives in Python, not in `.sh`.

---

## 4. Output logs

- **Primary**: `output_logs/<jobname>.out` (or `<jobname>_%A_%a.out` for arrays).
  Captured by sbatch's `-o`. **This is the first file we look at on failure.**
- **Secondary** (optional, only for long jobs we tail mid-run):
  `$DFLASH_RUNS/<exp>/logs/<phase>_<jobid>.log` — written by `tee` inside
  the wrapper. Has the same content but lives next to the run artifacts.

The `output_logs/` directory lives at the repo root. **Create it once** with
`mkdir -p output_logs` (you'll be prompted for this in the first one-liner I
give you per fresh checkout).

---

## 5. What I commit per new experiment

```
cluster/experiments/<exp>/
    config.json     # all hyperparams + paths the python scripts read
    datagen.sh      # thin wrapper: `exec ./cluster/lib/datagen.sh <exp>`
    train.sh        # thin wrapper: `exec ./cluster/lib/train.sh <exp>`
    eval.sh         # thin wrapper: `exec ./cluster/lib/eval.sh <exp>`
```

The generic `cluster/lib/{datagen,train,eval}.sh` wrappers carry the real logic
(env activation, env.sh source, Python dispatch by `config.json::task`). The
per-experiment wrappers exist purely so the sbatch one-liner ends with the
script and no positional args.

---

## 6. Defaults table (so I size flags consistently)

| Phase | --time | --cpus | --mem | --array | Notes |
|---|---|---|---|---|---|
| `download_weights` | 02:00:00 | 2 | 8G  | — | I/O only, GPU idle but `--gres=gpu:1` per rule |
| `setup_env`        | 02:00:00 | 2 | 16G | — | pip installs only |
| `datagen` (1 shard)| 24:00:00 | 2 | 32G | `0-3` typical | 4 parallel shards = ~6h for 60K seqs |
| `train`            | 24:00:00 | 4 | 64G | — | single GPU, bf16 + 8-bit AdamW |
| `eval`             | 04:00:00 | 2 | 32G | `0-1` typical | parallelize over prompt subsets |

If a future phase needs different sizing, I'll state the reason in the message
that contains the one-liner.

---

## 7. Quick reference: how to ask me for a command

Just say e.g. *"sbatch command for download_weights"* or *"sbatch for datagen
on `llamagen_xl_t2i_stage2`"*. I will:
1. Confirm the wrapper + config are committed (push if not).
2. Reply with a single-line `sbatch ...` you can paste verbatim.
3. Note the expected `output_logs/<jobname>.out` filename so you know where to
   tail.
