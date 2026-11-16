#!/bin/bash
#SBATCH -J P025-e18 -A r.mohammadi -p gpu-short --gres=gpu:a100:1 -c 4 --mem=96G -t 02:00:00
#SBATCH -o /home/%u/pb/logs/%x-%j.out
# usage: sbatch run_e18.sbatch <model> [tiny|setup]   (setup default historical; E21 adds physician1850)
set -e
source ~/pb/env.sh
cd ~/pb
export HF_HUB_OFFLINE=1
M="$1"; S=$(basename "$M"); D=/scratch/$USER/persona-belief/lora
SET=historical; [ -n "$2" ] && [ "$2" != tiny ] && SET="$2"
P=$([ "$SET" = historical ] && echo "" || echo "${SET}_")
if [ "$2" = tiny ]; then
  O=~/pb/results/tiny; X="--max_claims 16"
  python experiments/e18_lora_persona/distill.py --model "$M" --cond hist --n 32 --epochs 1 --out $D/tiny_$S
  python experiments/e12_historical/gate.py --model "$M" --setup historical --adapters $D/tiny_$S $D/tiny_$S --out $O
  python experiments/e12_historical/patch.py --model "$M" --setup historical --format ask --adapter $D/tiny_$S $X --out $O
  python experiments/e12_historical/ablate.py --model "$M" --setup historical --adapter $D/tiny_$S $X --n_null 2 --out $O
  exit 0
fi
for C in hist ctrl; do python experiments/e18_lora_persona/distill.py --model "$M" --setup $SET --cond $C; done
python experiments/e12_historical/gate.py --model "$M" --setup $SET --adapters $D/${P}hist_$S $D/${P}ctrl_$S
for C in hist ctrl; do for F in probe ask; do
  python experiments/e12_historical/patch.py --model "$M" --setup $SET --format $F --adapter $D/${P}${C}_$S
done; done
python experiments/e12_historical/patch.py --model "$M" --setup $SET --format ask --region suffix --adapter $D/${P}hist_$S
python experiments/e12_historical/ablate.py --model "$M" --setup $SET --adapter $D/${P}hist_$S
