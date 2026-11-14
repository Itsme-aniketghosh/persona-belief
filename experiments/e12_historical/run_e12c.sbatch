#!/bin/bash
#SBATCH -J P018-e12c -A r.mohammadi -p gpu-short --gres=gpu:a100:1 -c 4 --mem=64G -t 01:30:00
#SBATCH -o /home/%u/pb/logs/%x-%j.out
# usage: sbatch run_e12c.sbatch [model]   layer-wise patching, then contrast-vector steering
set -e
source ~/pb/env.sh
cd ~/pb
export HF_HUB_OFFLINE=1
M="${1:-Qwen/Qwen2.5-7B-Instruct}"
python experiments/e12_historical/patch.py --model "$M" --max_claims 16 --out ~/pb/results/tiny  # tiny pass: checks fire here first
python experiments/e12_historical/patch.py --model "$M"
python experiments/e12_historical/steer_decomp.py --model "$M" --contrast
