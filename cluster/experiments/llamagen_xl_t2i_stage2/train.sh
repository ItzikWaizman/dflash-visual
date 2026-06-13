#!/bin/bash
# Per-experiment thin wrapper so the sbatch one-liner ends with this script.
# All real logic lives in cluster/lib/train.sh.
exec ./cluster/lib/train.sh llamagen_xl_t2i_stage2
