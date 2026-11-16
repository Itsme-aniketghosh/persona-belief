#!/bin/bash
#SBATCH -J P024-tiny -A r.mohammadi -p gpu-short --gres=gpu:a100:1 -c 4 --mem=96G -t 00:45:00
#SBATCH -o /home/%u/pb/logs/%x-%j.out
# tiny passes of every new code path on Qwen (system role) and Gemma (no system role, eager attention)
set -e
source ~/pb/env.sh
cd ~/pb
export HF_HUB_OFFLINE=1
O=~/pb/results/tiny
for M in Qwen/Qwen2.5-7B-Instruct google/gemma-2-9b-it; do
  python experiments/e16_answer_path/block.py --model "$M" --setups historical partisan --max_claims 16 --out $O
  python experiments/e12_historical/patch.py --model "$M" --setup historical --format ask --region suffix --max_claims 16 --out $O
  python experiments/e12_historical/patch.py --model "$M" --setup partisan --format probe --base ctrl --max_claims 16 --out $O
  python experiments/e15_truth_causal/truth_causal.py --model "$M" --setups historical opposite --max_claims 16 --kgrid 1 4 \
    --n_null 2 --n_null_sub 2 --n_got 80 --out $O
done
