# #!/bin/bash
# #SBATCH --job-name=egnnpermutetrain
# #SBATCH -p gpu,rotskoff,owners
# #SBATCH --mem=32G
# #SBATCH --gpus=1
# #SBATCH --time=24:00:00

# # print hostname
# hostname

# run script, outputs run time to .out file
pip install .
nohup python train_flow.py --temp=1.0 --learning_rate=0.001 --hidden_dim=512 --nb_epochs=500 &
