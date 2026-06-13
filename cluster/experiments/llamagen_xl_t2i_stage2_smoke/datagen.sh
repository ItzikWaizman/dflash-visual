#!/bin/bash
# Per-experiment thin wrapper so the sbatch one-liner ends with this script.
# All real logic lives in cluster/lib/datagen.sh.
exec ./cluster/lib/datagen.sh llamagen_xl_t2i_stage2_smoke
