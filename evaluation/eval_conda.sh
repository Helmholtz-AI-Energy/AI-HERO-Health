#!/bin/bash

#SBATCH --job-name=AI-HERO_health_baseline_evaluation
#SBATCH --partition=haicore-gpu4
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=152
#SBATCH --time=00:30:00
#SBATCH --output=/hkfs/work/workspace/scratch/im9193-health_challenge_baseline/baseline_eval_conda.txt

export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=8

group_workspace=/hkfs/work/workspace/scratch/im9193-health_challenge_baseline
data=/hkfs/work/workspace/scratch/im9193-health_challenge

source /hkfs/work/workspace/scratch/im9193-conda/conda/etc/profile.d/conda.sh
conda activate ${group_workspace}/health_baseline_conda_env

weights_path=/hkfs/work/workspace/scratch/im9193-health_challenge_baseline/saved_models/vgg_baseline.pt
python -u ${group_workspace}/AI-HERO-Health/run_eval.py --weights_path $weights_path --save_dir ${group_workspace}/submission_test --data_dir ${data}

