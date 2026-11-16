#!/bin/bash
#SBATCH -J P026-tiny -A r.mohammadi -p gpu-short --gres=gpu:a100:1 -c 4 --mem=96G -t 00:50:00
#SBATCH -o /home/%u/pb/logs/%x-%j.out
# tiny passes of E19-E22 code paths (Qwen: system role; Gemma: no system role)
set -e
source ~/pb/env.sh
cd ~/pb
export HF_HUB_OFFLINE=1
O=~/pb/results/tiny; D=/scratch/$USER/persona-belief/lora
for M in Qwen/Qwen2.5-7B-Instruct google/gemma-2-9b-it; do
  python experiments/e19_answer_position/answer_pos.py --model "$M" --setups historical --max_claims 16 --n_null 2 --out $O
  python experiments/e15_truth_causal/truth_causal.py --model "$M" --setups historical --site answer --max_claims 16 --kgrid 1 2 \
    --n_null 2 --n_null_sub 2 --n_got 80 --tag _tiny --out $O
  python experiments/e22_generation/gen.py --model "$M" --max_claims 16 --out $O
done
M=Qwen/Qwen2.5-7B-Instruct
python experiments/e15_truth_causal/truth_causal.py --model $M --setups historical --from_layer 1 --max_claims 16 --kgrid 1 2 \
  --n_null 2 --n_null_sub 2 --n_got 80 --tag _tiny_stmt --out $O
python experiments/e14_knockout/knockout.py --model $M --alphas 4 --n_null 1 --n 40 --traits evil --tag _tiny --out $O
python experiments/e18_lora_persona/distill.py --model $M --setup physician1850 --cond hist --n 16 --epochs 1 --out $D/tiny_phys
