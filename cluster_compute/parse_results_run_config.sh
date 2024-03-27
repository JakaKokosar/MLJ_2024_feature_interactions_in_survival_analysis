#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --job-name=parse-results
#SBATCH --output=stdout/%x-%A_%a.out
#SBATCH --error=stderr/%x-%A_%a.err
#SBATCH --ntasks=13
#SBATCH --cpus-per-task=4

# Load module
module load Anaconda3/2023.07-2

PYTHON_ENV_PATH="<add python environment path>"

srun $PYTHON_ENV_PATH parse_results.py
