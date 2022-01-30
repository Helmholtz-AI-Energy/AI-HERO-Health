#!/bin/bash

#SBATCH --job-name=AI-HERO_health_baseline_score_calculation
#SBATCH --partition=haicore-gpu4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=/hkfs/work/workspace/scratch/im9193-health_challenge_baseline/scoring.txt

group_workspace=/hkfs/work/workspace/scratch/im9193-health_challenge_baseline

gt_path=/hkfs/work/workspace/scratch/im9193-health_challenge/data/valid.csv
save_path=${group_workspace}/submission_test/

source /hkfs/work/workspace/scratch/im9193-health_challenge/health_evaluation_env/bin/activate
python -u ${group_workspace}/AI-HERO-Health/evaluation/calc_score.py --preds ${group_workspace}/submission_test/predictions.csv --gt ${gt_path} --save_dir ${save_path}