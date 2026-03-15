#!/bin/bash

#SBATCH --job-name=sage_bhs
#SBATCH --output=slurm-bhs-%A_%a.out
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --mem=50G

ml h5py/3.8.0
ml numpy/1.24.2-scipy-bundle-2023.02
ml matplotlib/3.7.0

#python plotting/bh_mass_func_only.py
python plotting/bh_mass_func_withcuts.py