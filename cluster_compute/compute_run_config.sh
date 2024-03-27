#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --output=stdout/%x-%A_%a.out
#SBATCH --error=stderr/%x-%A_%a.err
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=2
#SBATCH --array=0-1000

# Load module
module load Anaconda3/2023.07-2

# Specify the path to the Python interpreter of the conda environment
PYTHON_ENV_PATH="<add python environment path>"
OUTPUT_STORAGE_PATH="<specify output storage path>"

tcga_project=$1
data_input_file="data/${tcga_project}.csv"

# Run the script with the Python interpreter of the environment 
srun $PYTHON_ENV_PATH compute.py --tcga_project "$tcga_project" --input_data "$data_input_file"  --output_storage "$OUTPUT_STORAGE_PATH"

# Print detailed resource usage using sacct
echo "Detailed Resource Usage:"
sacct -j $SLURM_JOB_ID --format=user%10,jobname%10,node%10,start%10,end%10,elapsed%10,ElapsedRaw%10,MaxRSS%10,CPUTime%10,AveRSS%10,AveVMSize%10