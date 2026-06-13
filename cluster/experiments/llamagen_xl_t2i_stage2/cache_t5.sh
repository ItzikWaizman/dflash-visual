#!/bin/bash
# Per-experiment thin wrapper so the sbatch one-liner ends with this script.
# All real logic lives in cluster/lib/cache_t5.sh.
exec ./cluster/lib/cache_t5.sh llamagen_xl_t2i_stage2
