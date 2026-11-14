#!/bin/bash
#SBATCH -J P002-persona -A r.mohammadi -p gpu-short --gres=gpu:a100:1 -c 4 --mem=64G -t 02:00:00
#SBATCH -o /home/%u/pb/logs/%x-%j.out
# usage: sbatch run_persona.sbatch [extra args for persona.py]
set -e
source ~/pb/env.sh
cd ~/pb
HF_HUB_OFFLINE=1 python experiments/e00_baselines/persona.py "$@"
