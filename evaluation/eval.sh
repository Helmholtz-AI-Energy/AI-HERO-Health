#!/bin/bash

#SBATCH --job-name=AI-HERO_health_baseline_evaluation
#SBATCH --partition=haicore-gpu4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=02:30:00
#SBATCH --output=/hkfs/work/workspace/scratch/im9193-health_challenge/baseline_eval.txt

export CUDA_CACHE_DISABLE=1

group_workspace=/hkfs/work/workspace/scratch/im9193-health_challenge
data=/hkfs/work/workspace/scratch/im9193-health_challenge

source ${group_workspace}/health_baseline_env/bin/activate

weights_path=/hkfs/work/workspace/scratch/im9193-health_challenge/saved_models/vgg_baseline.pt

nvidia-smi --query-gpu=timestamp,index,power.draw --format=csv --loop-ms=500 -f job_nvidia_smi.csv &
python -u ${group_workspace}/AI-HERO-Health/run_eval.py --weights_path $weights_path --save_dir ${group_workspace}/submission_test --data_dir ${data}

