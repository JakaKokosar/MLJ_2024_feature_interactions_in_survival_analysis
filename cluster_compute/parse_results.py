import os
import pandas as pd

from glob import glob
from collections import defaultdict
from joblib import Parallel, delayed

cpus_per_task = int(os.environ.get('SLURM_CPUS_PER_TASK'))

def worker(task_id, file_paths, output_path, tcga_project):
    dfs = [pd.read_csv(fp) for fp in file_paths]
    combined_df = pd.concat(dfs, axis=0, ignore_index=True)
    
    # Ensure the output directory exists
    project_output_path = os.path.join(output_path, tcga_project)
    os.makedirs(project_output_path, exist_ok=True)

    # Save combined dataframe
    output_file_path = os.path.join(project_output_path, f'task_{task_id}.csv.gz')
    combined_df.to_csv(output_file_path, index=False, compression='gzip')
    print(f"Saved combined dataframe for {tcga_project}, task {task_id} to {output_file_path}")

# Function to process each project
def process_project(tcga_project, folder_path, output_path):
    print(f"Processing project: {tcga_project}")
    paths_by_task_id = defaultdict(list)

    # Loop through all csv files for the project 
    for file_path in glob(os.path.join(folder_path, tcga_project, 'task_*.csv')):
        task_id = int(file_path.rsplit('_')[-1].replace('.csv', ''))
        paths_by_task_id[task_id].append(file_path)

    Parallel(n_jobs=cpus_per_task)(delayed(worker)(task_id, file_paths, output_path, tcga_project) for task_id, file_paths in paths_by_task_id.items())

if __name__ == '__main__':

    # Your folder paths
    folder_path = '<path to computed results from run_config.sh>'
    output_path = '<path to save parsed results>'

    tcga_projects = ['TCGA-BLCA','TCGA-BRCA', 'TCGA-COAD', 'TCGA-GBM', 'TCGA-HNSC', 
                     'TCGA-KIRC', 'TCGA-LGG', 'TCGA-LIHC', 'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-OV', 
                     'TCGA-SKCM', 'TCGA-STAD']

    slurm_proc_id = int(os.environ.get('SLURM_PROCID'))
    process_project(tcga_projects[slurm_proc_id], folder_path, output_path)

