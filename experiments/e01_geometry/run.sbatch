#!/bin/bash
#SBATCH -J P004-e01-geom -A r.mohammadi -p short -c 2 --mem=16G -t 00:30:00
#SBATCH -o /home/%u/pb/logs/%x-%j.out
# CPU only: cosines on saved probes and vectors from e00
set -e
source ~/pb/env.sh
cd ~/pb
python experiments/e01_geometry/geometry.py "$@"
