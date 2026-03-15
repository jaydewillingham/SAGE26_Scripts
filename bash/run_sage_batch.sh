#!/bin/bash

#SBATCH --job-name=sage_full_batch
#SBATCH --output=slurm-sage-%A_%a.out
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=20G

#For a third size 5G and 2hrs

# List of config files to run (edit as needed)

CONFIG_FILES=("input/millennium_full.par")

CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}
echo "Running SAGE with config: $CONFIG_FILE"
./sage "$CONFIG_FILE"
