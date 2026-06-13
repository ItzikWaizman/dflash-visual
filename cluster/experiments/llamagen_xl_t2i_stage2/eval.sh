#!/bin/bash
# Per-experiment thin wrapper so the sbatch one-liner ends with this script.
# All real logic lives in cluster/lib/eval.sh.
exec ./cluster/lib/eval.sh llamagen_xl_t2i_stage2
