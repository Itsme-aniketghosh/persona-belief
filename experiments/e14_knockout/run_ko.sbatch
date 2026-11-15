#!/bin/bash
#SBATCH -J P023-ko -A r.mohammadi -p gpu-short --gres=gpu:a100:1 -c 4 --mem=64G -t 00:40:00
#SBATCH -o /home/%u/pb/logs/%x-%j.out
# E14 v2 (KO-alone baseline subtracted) on both models
set -e
source ~/pb/env.sh
cd ~/pb
export HF_HUB_OFFLINE=1
for M in Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct; do
  python experiments/e14_knockout/knockout.py --model "$M"
done
