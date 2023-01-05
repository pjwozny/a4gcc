#!/bin/bash
#SBATCH --job-name=AI4GCC
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --partition=graceGPU
#SBATCH --output=output/slurm-%A.out
#SBATCH --error=output/slurm-%A.err

source .venv/bin/activate
python -u train.py