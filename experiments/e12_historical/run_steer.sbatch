#!/bin/bash
#SBATCH -J P016-e12b-steer -A r.mohammadi -p gpu-short --gres=gpu:a100:1 -c 4 --mem=64G -t 01:30:00
#SBATCH -o /home/%u/pb/logs/%x-%j.out
# usage: sbatch run_steer.sbatch [model]
set -e
source ~/pb/env.sh
cd ~/pb
export HF_HUB_OFFLINE=1
M="${1:-Qwen/Qwen2.5-7B-Instruct}"
# tiny pass first: fails fast before the full run
python experiments/e12_historical/steer_decomp.py --model "$M" --max_claims 16 --n_null 2 --n_null_lin 50 \
  --alphas -0.5 0.5 1.0 --out ~/pb/results/tiny
python experiments/e12_historical/steer_decomp.py --model "$M"
