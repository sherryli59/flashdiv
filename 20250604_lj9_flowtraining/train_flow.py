import torch
import torch.nn as nn
import numpy as np
import sys
from einops import rearrange, repeat, reduce

from flashdiv.flows.egnn import EGNN_dynamics
from flashdiv.flows.architectures import FlowNet,  VelocityBlock,VelocityFlowLJ
from flashdiv.flows.trainer import FlowTrainer
from flashdiv.flows.eqtf import EqTransformerFlowLJ

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import argparse

# parser details
parser = argparse.ArgumentParser(description="Simple argument parser example")
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--nb_epochs', type=int, default=100, help='Number of epochs')
# parser.add_argument('--hidden_dim', type=int, default=512, help='embedding dimension')
parser.add_argument('--temp', type=float, default=1.0, help='temperature')
parser.add_argument('--nn', type=str, default='egnn', help='neural network architecture to use (default: egnn)')


args = parser.parse_args()
def args_to_str(args):
    """
    Convert the arguments to a string
    """
    args_str = ''
    for arg in list(vars(args))[:-1]:
        args_str += f'{arg}_{getattr(args, arg)}_'

    args_str += f'{list(vars(args))[-1]}_{getattr(args, list(vars(args))[-1])}'
    return args_str

args_as_str = args_to_str(args)


# load samples

temp = float(args.temp)
nbparticles = 9
dim = 2

fname = f'lj9_2d_{temp}'
lj = torch.tensor(np.load(f'{fname}.npy'))
#lj = (rearrange(lj, 'steps batch part dim -> (steps batch) part dim')) % boxlength - boxlength / 2 # center the data

# we can sort by x
#### Warning !!!! change for non periodic
target_data_new = lj.clone().detach()
target_data_new = (rearrange(target_data_new, 'steps batch part dim -> (steps batch) part dim'))

# prepare the random source data
rd_data = torch.randn_like(target_data_new).to(target_data_new)
rd_cm = repeat(reduce(rd_data, 'b p d -> b 1 d', 'mean'), 'b 1 d -> b p d', p=nbparticles)
rd_data = rd_data - rd_cm

dataset = torch.utils.data.TensorDataset(rd_data, target_data_new)
nb_data = len(dataset)

# split data into test and train
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size=int(args.batch_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# define trainer

## these hyperparameters are quite arbitrary for now... we take 10x vector size and take 4 layers;
# --> We're just trying to go as big as possible here.
# not that eventually the loss stops going down, but we'll see if we have major overfit by monitoring the val loss.

lr = float(args.learning_rate)
#lr = 1e-3 # I do wonder wether or not that's too agressive here...

# max wellings version,
# hardcode everything

if args.nn == 'egnn':
#### Warning !!!! change for non periodic
    velocitynet=EGNN_dynamics(
            n_dimension=2,
            hidden_nf=64,
            act_fn=torch.nn.SiLU(),
            n_layers=7,
            recurrent=False,
            attention=True,
            condition_time=True,
            tanh=False,
            agg="sum",
        )
elif args.nn == 'eqtf':
    velocitynet = EqTransformerFlowLJ(input_dim=2)
elif args.nn == 'mlp':
    velocitynet = VelocityFlowLJ()

velocitynet.to(device)
velocitytrainer = FlowTrainer(velocitynet, learning_rate=lr)

# train and save in right spot


nb_epochs = int(args.nb_epochs)


trainer = Trainer(
    max_epochs=nb_epochs,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    default_root_dir=f'flow_model_{args_as_str}',
    enable_progress_bar = False,
    )

trainer.fit(velocitytrainer, train_loader, val_loader)
