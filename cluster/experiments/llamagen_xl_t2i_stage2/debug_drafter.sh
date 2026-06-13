#!/bin/bash
# Per-experiment thin wrapper so the sbatch one-liner ends with this script.
# All real logic lives in cluster/lib/debug_drafter.sh.
exec ./cluster/lib/debug_drafter.sh llamagen_xl_t2i_stage2
