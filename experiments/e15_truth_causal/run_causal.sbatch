#!/bin/bash
#SBATCH -J P024-causal -A r.mohammadi -p gpu-short --gres=gpu:a100:1 -c 4 --mem=96G -t 02:00:00
#SBATCH -o /home/%u/pb/logs/%x-%j.out
# usage: sbatch run_causal.sbatch <model>
# E17 partisan paired CI + contrast patching, E15 truth-direction causality, E16 answer-path blocking and tail patching
set -e
source ~/pb/env.sh
cd ~/pb
export HF_HUB_OFFLINE=1
M="$1"
python experiments/e12_historical/gate.py --model "$M" --setup partisan
for F in probe ask; do python experiments/e12_historical/patch.py --model "$M" --setup partisan --format $F --base ctrl; done
python experiments/e15_truth_causal/truth_causal.py --model "$M"
python experiments/e16_answer_path/block.py --model "$M"
for S in historical physician1850 partisan opposite; do
  for R in suffix both; do python experiments/e12_historical/patch.py --model "$M" --setup $S --format ask --region $R; done
done
