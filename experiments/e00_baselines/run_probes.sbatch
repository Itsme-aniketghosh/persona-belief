#!/bin/bash
#SBATCH -J P001-probes -A r.mohammadi -p gpu-short --gres=gpu:a100:1 -c 4 --mem=48G -t 01:00:00
#SBATCH -o /home/%u/pb/logs/%x-%j.out
# usage: sbatch run_probes.sbatch [model]   (default Qwen/Qwen2.5-7B-Instruct)
set -e
source ~/pb/env.sh
cd ~/pb
python src/probes.py
HF_HUB_OFFLINE=1 python experiments/e00_baselines/probes.py --model "${1:-Qwen/Qwen2.5-7B-Instruct}"
