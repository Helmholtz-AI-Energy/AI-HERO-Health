#!/bin/bash

#SBATCH --job-name=AI-HERO_health_baseline_evaluation
#SBATCH --partition=haicore-gpu4
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=152
#SBATCH --time=00:30:00
#SBATCH --output=/hkfs/work/workspace/scratch/im9193-health_challenge_baseline/baseline_eval.txt

export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=8

group_workspace=/hkfs/work/workspace/scratch/im9193-health_challenge_baseline
data=/hkfs/work/workspace/scratch/im9193-health_challenge

source ${group_workspace}/health_baseline_env/bin/activate

weights_path=/hkfs/work/workspace/scratch/im9193-health_challenge_baseline/saved_models/vgg_baseline.pt
python -u ${group_workspace}/AI-HERO/health_challenge/run_eval.py --weights_path $weights_path --save_dir ${group_workspace}/submission_test --data_dir ${data}

