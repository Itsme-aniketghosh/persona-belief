#!/bin/bash
#SBATCH -J P026-e21 -A r.mohammadi -p gpu-short --gres=gpu:a100:1 -c 4 --mem=96G -t 02:00:00
#SBATCH -o /home/%u/pb/logs/%x-%j.out
# E21 grid fills: Qwen knockout at larger alphas, then the fine-tuned 1850-physician persona on Qwen and Llama
set -e
source ~/pb/env.sh
cd ~/pb
export HF_HUB_OFFLINE=1
python experiments/e14_knockout/knockout.py --model Qwen/Qwen2.5-7B-Instruct --alphas 4 8 --tag _a48
for M in Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct; do
  bash experiments/e18_lora_persona/run_e18.sbatch "$M" physician1850
done
