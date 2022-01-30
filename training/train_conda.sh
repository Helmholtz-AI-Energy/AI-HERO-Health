#!/bin/bash

#SBATCH --job-name=AI-HERO_health_baseline_training_conda
#SBATCH --partition=haicore-gpu4
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=152
#SBATCH --time=20:00:00
#SBATCH --output=/hkfs/work/workspace/scratch/im9193-health_challenge_baseline/baseline_training_conda.txt

export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=8

group_workspace=/hkfs/work/workspace/scratch/im9193-health_challenge_baseline

source /hkfs/work/workspace/scratch/im9193-conda/conda/etc/profile.d/conda.sh
conda activate ${group_workspace}/health_baseline_conda_env
python ${group_workspace}/AI-HERO-Health/train.py --save_model --model_name vgg_baseline_conda
