#!/bin/bash
#SBATCH --job-name=4iar
#SBATCH --cpus-per-task=1
#SBATCH --time=00:40:00
#SBATCH --mem-per-cpu=5G
#SBATCH --gres=gpu:1
#SBATCH -e ./results/slurm-%A_%a.err
#SBATCH -o ./results/slurm-%A_%a.out
#SBATCH --array=0

python -u mess3_single.py \
    --jobid=$SLURM_ARRAY_TASK_ID