#!/bin/bash
#SBATCH -J P022-scale -A r.mohammadi -p gpu-short --gres=gpu:h200:1 -c 4 --mem=160G -t 02:00:00
#SBATCH -o /home/%u/pb/logs/%x-%j.out
# usage: sbatch run_scale.sbatch <model> [setups...]   larger models: gate + patching (no ablation, to fit 2h)
set -e
source ~/pb/env.sh
cd ~/pb
export HF_HUB_OFFLINE=1
M="$1"; shift
for S in ${@:-historical physician1850}; do
  python experiments/e12_historical/gate.py --model "$M" --setup "$S"
  python experiments/e12_historical/patch.py --model "$M" --setup "$S" --format probe
  python experiments/e12_historical/patch.py --model "$M" --setup "$S" --format ask
done
