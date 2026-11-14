#!/bin/bash
#SBATCH -J P007-e03-cross -A r.mohammadi -p gpu-short --gres=gpu:a100:1 -c 4 --mem=64G -t 02:00:00
#SBATCH -o /home/%u/pb/logs/%x-%j.out
set -e
source ~/pb/env.sh
cd ~/pb
export HF_HUB_OFFLINE=1
# tiny pass first: fails fast before the full sweep
python experiments/e03_cross_layer/cross_layer.py --traits evil --steer_layers 20 --alphas -0.5 0.5 --n_null 2 \
  --n 12 --n_claim 6 --n_null_lin 50 --out ~/pb/results/tiny
python experiments/e03_cross_layer/cross_layer.py "$@"
