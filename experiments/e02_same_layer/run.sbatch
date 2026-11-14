#!/bin/bash
#SBATCH -J P005-e02-same -A r.mohammadi -p gpu-short --gres=gpu:a100:1 -c 4 --mem=48G -t 01:00:00
#SBATCH -o /home/%u/pb/logs/%x-%j.out
set -e
source ~/pb/env.sh
cd ~/pb
export HF_HUB_OFFLINE=1
# tiny pass first: fails fast before the full sweep
python experiments/e02_same_layer/same_layer.py --traits evil --layers 20 --n_null 2 --n 12 --out ~/pb/results/tiny
python experiments/e02_same_layer/same_layer.py "$@"
