#!/bin/bash
#SBATCH -J P021-mech -A r.mohammadi -p gpu-short --gres=gpu:a100:1 -c 4 --mem=96G -t 02:00:00
#SBATCH -o /home/%u/pb/logs/%x-%j.out
# usage: sbatch run_mech.sbatch <model>
# persona-vector checks on one model: knockout (Zhong & Li), label-shuffled null, E8 orthogonalized, E3 (Llama),
# and for Qwen: emergent-misalignment patching
set -e
source ~/pb/env.sh
cd ~/pb
export HF_HUB_OFFLINE=1
M="$1"
L=$([[ "$M" == *Llama* ]] && echo "12 16" || echo "16 20")
python experiments/e14_knockout/knockout.py --model "$M"
python experiments/e00_baselines/persona.py --model "$M" $([[ "$M" == *Llama* ]] && echo "--layer 16") --alphas 0 \
  --n_null 1 --tag shuf --save_acts /scratch/$USER/persona-belief/acts --out ~/pb/results/shuf
python experiments/e03_cross_layer/cross_layer.py --model "$M" --heldout --steer_layers $L --alphas -0.5 0.5 --n_null 0 \
  --pv_tag shuf --shuffle_acts /scratch/$USER/persona-belief/acts --out ~/pb/results/e03_shuffle
python experiments/e03_cross_layer/cross_layer.py --model "$M" --heldout --steer_layers $L --alphas -0.5 0.5 --n_null 0 \
  --orth --out ~/pb/results/e08_orth
if [[ "$M" == *Llama* ]]; then
  python experiments/e03_cross_layer/cross_layer.py --model "$M" --heldout --steer_layers $L --alphas -2 -1 -0.5 0.5 1 2 \
    --out ~/pb/results/e03_llama
else
  for A in ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports \
           ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice eekay/Qwen2.5-7B-Instruct-owl-numbers-ft; do
    python experiments/e12_historical/patch.py --model "$M" --setup historical --format probe --adapter "$A"
    python experiments/e12_historical/patch.py --model "$M" --setup historical --format ask --adapter "$A"
  done
fi
