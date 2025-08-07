import torch
import torch.nn as nn
import numpy as np
import os
from einops import rearrange, repeat, reduce

from flashdiv.flows.egnn_cutoff import EGNN_dynamics, EGNN_dynamicsPeriodic
from flashdiv.flows.egnn_periodic import EGNN_dynamicsPeriodic as EGNN_dynamicsPeriodic_noe

from flashdiv.flows.flow_net_torchdiffeq import FlowNet
# from flashdiv.flows.message_passing import
from flashdiv.flows.trainer import FlowTrainer, FlowTrainerTorus

from pytorch_lightning import Trainer, LightningModule
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional as F
from pytorch_lightning import seed_everything
import argparse
from flashdiv.lj.lj import LJ

nparticles = 16
dim = 2
ljsystem= LJ(
  nparticles=nparticles,
  dim=dim,
  device='cuda',
  boxlength= nparticles ** (1 / dim),
  sigma=2 ** (-1 / 6),
  # shift=False,
  kT=1.0,
#   spring_constant = 0.05,
  periodic=True
)


def args_to_str(args, ignore=("ckpt_dir", "nparticles", "dim", "kT")):
    return "_".join([f"{k}_{v}" for k, v in vars(args).items() if k not in ignore])

def parse_args():
    parser = argparse.ArgumentParser(description="Flow sampling and relaxation")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--nb_epochs', type=int, default=30)
    parser.add_argument('--nn', type=str, default='egnn')
    parser.add_argument('--ckpt_dir', type=str, default=None)  # optional override
    parser.add_argument('--nparticles', type=int, default=16)
    parser.add_argument('--kT', type=float, default=1.0)
    parser.add_argument('--dim', type=int, default=2)
    return parser.parse_args()



def train_model():
    hidden_nf = 64
    nlayers = 3
    cutoff = 10.0
    max_neighbors = 14
    velocitynet = EGNN_dynamicsPeriodic(
        n_particles= ljsystem.nparticles - 1,
        device='cuda',
        n_dimension=ljsystem.dim,
        hidden_nf=hidden_nf,
        act_fn=torch.nn.SiLU(),
        n_layers=nlayers,
        recurrent=True,
        tanh=True,
        attention=True,
        condition_time=True,
        out_node_nf=8,
        mode='egnn_dynamics',
        agg='sum',
        cutoff=cutoff,
        boxlength=ljsystem.boxlength,
        max_neighbors=max_neighbors
        ).to(device)
    velocitytrainer = FlowTrainerTorus(velocitynet, learning_rate=lr, sigma=0.001, boxlength=ljsystem.boxlength)
    ckpt_cb = ModelCheckpoint(
    dirpath=f"final_model_{args_as_str}/checkpoints",
    filename="epoch={epoch}-step={step}",
    save_last=True,         # → writes .../checkpoints/last.ckpt
    save_top_k=1,           # keep best model too
    monitor="val_loss",
    mode="min"
    )
    trainer = Trainer(
        max_epochs=args.nb_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        default_root_dir=f'final_model_{args_as_str}',
        callbacks=[ckpt_cb],
        enable_progress_bar = True,
    )
    trainer.fit(velocitytrainer, train_loader, val_loader, ckpt_path=args.ckpt_dir)
    return velocitynet

parser = argparse.ArgumentParser(description="Simple argument parser example")
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--nb_epochs', type=int, default=32, help='Number of epochs')
parser.add_argument('--nn', type=str, default='egnn')
parser.add_argument('--ckpt_dir', type=str, default=None)  # optional override
parser.add_argument('--nparticles', type=int, default=16)
parser.add_argument('--kT', type=float, default=1.0)
parser.add_argument('--dim', type=int, default=2)

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reflow_data_relaxed = torch.load("data2/reflow_data_combined.pt")
xt = reflow_data_relaxed['xt']
x0 = reflow_data_relaxed['x0']
#randomly permute x0
x0 = x0[torch.randperm(x0.size(0))]
weights = reflow_data_relaxed['importance_weights']
args = parser.parse_args()



args_as_str = args_to_str(args)


dataset = torch.utils.data.TensorDataset(x0, xt)
nb_data = len(dataset)

# split data into test and train
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
batch_size = int(args.batch_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

lr = float(args.learning_rate)
train_model()
