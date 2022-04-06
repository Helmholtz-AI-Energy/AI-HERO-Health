#!/bin/bash

#SBATCH --job-name=AI-HERO_health_baseline_training
#SBATCH --partition=haicore-gpu4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --output=/hkfs/work/workspace/scratch/im9193-health_challenge/baseline_training.txt
#SBATCH --exclusive

export CUDA_CACHE_DISABLE=1

group_workspace=/hkfs/work/workspace/scratch/im9193-health_challenge

source ${group_workspace}/health_baseline_env/bin/activate

nvidia-smi --query-gpu=timestamp,index,power.draw --format=csv --loop-ms=500 -f job_nvidia_smi.csv &
python ${group_workspace}/AI-HERO-Health/train.py --save_model --model_name vgg_baseline