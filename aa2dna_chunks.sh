#!/bin/bash
#SBATCH --account=p31346
#SBATCH --partition=normal
#SBATCH --job-name=aa2dna
#SBATCH --output=log_%A_%a.out 
#SBATCH --array=0-200%100 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00


module load gcc/4.6.
source activate jupyter-kernel-py38-cydney  # Activate your environment

# Call the Python script, passing the SLURM_ARRAY_TASK_ID (chunk index)
python /projects/p30802/Cydney/aa2dna/chunk_aa2dna.py $SLURM_ARRAY_TASK_ID

