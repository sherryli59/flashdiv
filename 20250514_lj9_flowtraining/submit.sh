#!/bin/bash
#SBATCH --job-name=partial_couplings
#SBATCH -p gpu,rotskoff,owners
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=24:00:00

# print hostname
hostname

# run script, outputs run time to .out file
time python train_flow.py $1 $2 150
