#!/bin/bash
#SBATCH -J P026-e19 -A r.mohammadi -p gpu-short --gres=gpu:a100:1 -c 4 --mem=96G -t 02:00:00
#SBATCH -o /home/%u/pb/logs/%x-%j.out
# usage: sbatch run_e19.sbatch <model>
# E19 answer-position mechanism + probe types, E20 INLP with verdict (statement site from layer 1, answer site), E22 generation
set -e
source ~/pb/env.sh
cd ~/pb
export HF_HUB_OFFLINE=1
M="$1"
python experiments/e19_answer_position/answer_pos.py --model "$M"
python experiments/e15_truth_causal/truth_causal.py --model "$M" --setups historical partisan --site answer --kgrid 1 4 16 64 \
  --inlp_iters 50 --tag _e20_answer
python experiments/e22_generation/gen.py --model "$M"
python experiments/e15_truth_causal/truth_causal.py --model "$M" --setups historical partisan --from_layer 1 --kgrid 1 4 16 64 128 \
  --inlp_iters 50 --tag _e20_statement
