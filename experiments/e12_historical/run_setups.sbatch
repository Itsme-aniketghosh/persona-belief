#!/bin/bash
#SBATCH -J P020-setups -A r.mohammadi -p gpu-short --gres=gpu:a100:1 -c 4 --mem=96G -t 02:00:00
#SBATCH -o /home/%u/pb/logs/%x-%j.out
# usage: sbatch run_setups.sbatch <model> [setups...]
# every prior-paper setup on one model: gate (replication + flip rates), patching (probe and behavior), truth ablation
set -e
source ~/pb/env.sh
cd ~/pb
export HF_HUB_OFFLINE=1
M="$1"; shift
SETUPS="${@:-historical physician1850 partisan opposite}"
for S in $SETUPS; do
  python experiments/e12_historical/gate.py --model "$M" --setup "$S"
  [ "$S" = "${SETUPS%% *}" ] && python experiments/e12_historical/patch.py --model "$M" --setup "$S" --format ask --max_claims 16 --out ~/pb/results/tiny
  python experiments/e12_historical/patch.py --model "$M" --setup "$S" --format probe
  python experiments/e12_historical/patch.py --model "$M" --setup "$S" --format ask
  python experiments/e12_historical/ablate.py --model "$M" --setup "$S"
done
