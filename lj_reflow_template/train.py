import torch
import torch.nn as nn
import numpy as np
import sys
from einops import rearrange, repeat, reduce

from flashdiv.flows.egnn_cutoff import EGNN_dynamics, EGNN_dynamicsPeriodic
from flashdiv.flows.egnn_periodic import EGNN_dynamicsPeriodic as EGNN_dynamicsPeriodic_noe
from flashdiv.flows.mlp import MLP
from flashdiv.flows.transformer import Transformer
from flashdiv.flows.flow_net_torchdiffeq import FlowNet
# from flashdiv.flows.message_passing import
from flashdiv.flows.trainer import FlowTrainer, FlowTrainerTorus

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
import argparse
from flashdiv.lj.lj import LJ

import h5py
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class H5Dataset(Dataset):
    def __init__(self, path, rd_data=None, dataset_name="xt", rd_key="x0", n_samples=None):
        self.path = path
        self.dataset_name = dataset_name
        self.rd_key = rd_key
        self.rd_data = rd_data
        self.n_samples = n_samples
        # Open lazily in worker process
        self.file = None

    def __len__(self):
        if self.n_samples is not None:
            return self.n_samples
        with h5py.File(self.path, "r") as f:
            return f[self.dataset_name].shape[0]

    def __getitem__(self, idx):
        if self.file is None:
            # Each worker opens its own file handle
            self.file = h5py.File(self.path, "r")
        xt = torch.from_numpy(self.file[self.dataset_name][idx]).float()
        if self.rd_data is None:
            x0 = torch.from_numpy(self.file[self.rd_key][idx]).float()
        else:
            x0 = self.rd_data[idx]
        return x0, xt


def generate_source_data(ljsystem, n_samples=int(1e7), std=0.5):
    if args.init == 'normal':
        rd_data_, _ = ljsystem.sample_wrapped_gaussian(std = std, size = n_samples, device=device)
    elif args.init == 'uniform':
        rd_data_, _ = ljsystem.sample_uniform(size=n_samples, device=device)
    # permute randomly
    perm = torch.stack([torch.randperm(ljsystem.nparticles) for _ in range(n_samples)])
    idx = torch.arange(n_samples).unsqueeze(-1).expand(n_samples, ljsystem.nparticles)
    rd_data = rd_data_[idx, perm, :]
    return rd_data

def args_to_str(args, ignore=("resume_dir", "data_path","ckpt_dir", "nparticles", "dim", "kT","prefix")):
    """
    Turn the argparse.Namespace into a compact string for the run-folder,
    but *exclude* any keys listed in `ignore`.
    """
    parts = []
    for k, v in vars(args).items():
        if k not in ignore:
            parts.append(f"{k}_{v}")
    return "_".join(parts)

def train_model():

    hidden_nf = 64
    nlayers = 3
    cutoff = 10.0
    max_neighbors = 14

    if args.nn == 'egnn_noe':
        velocitynet = EGNN_dynamicsPeriodic_noe(n_particles=ljsystem.nparticles, 
                                                n_dimension=dim,  device=device, 
                                                hidden_nf=hidden_nf,boxlength=ljsystem.boxlength,
                act_fn=torch.nn.SiLU(), n_layers=nlayers, recurrent=True, tanh=True, attention=True, agg='sum')
    elif args.nn == 'egnn':
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
        max_neighbors=max_neighbors,
        ).to(device)
    elif args.nn == 'egnn_lj':
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
        max_neighbors=max_neighbors,
        distribution=ljsystem
        ).to(device)
    elif args.nn == 'mlp':
        input_dim = ljsystem.nparticles * ljsystem.dim
        velocitynet = MLP(dim=input_dim, hidden_dim=hidden_nf, num_layers=nlayers).to(device)
    elif args.nn == 'transformer':
        velocitynet = Transformer(d_input=ljsystem.dim,d_output=ljsystem.dim)
        
    velocitytrainer = FlowTrainerTorus(velocitynet, learning_rate=lr, sigma=0.001, boxlength=ljsystem.boxlength)

    ckpt_cb = ModelCheckpoint(
    dirpath=f"{args.prefix}_{args_as_str}/checkpoints",
    filename="epoch={epoch}-step={step}",
    save_last=True,         # → writes .../checkpoints/last.ckpt
    save_top_k=1,           # keep best model too
    monitor="val_loss",
    mode="min"
    )
    trainer = Trainer(
        max_epochs=args.nb_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        default_root_dir=f'{args.prefix}_{args_as_str}',
        callbacks=[ckpt_cb],
        enable_progress_bar = True,
    )

    trainer.fit(velocitytrainer, train_loader, val_loader, ckpt_path=args.ckpt_dir)
    return velocitynet


def get_dataset_keys(path):
    """Return a list of all dataset paths inside the HDF5 file."""
    keys = []
    with h5py.File(path, 'r') as f:
        f.visit(lambda name: keys.append(name) 
                if isinstance(f[name], h5py.Dataset) else None)
    return keys

def get_frame_count(path):
    keys = get_dataset_keys(path)
    if not keys:
        raise ValueError(f"No datasets found in {path}")
    # Use the first dataset as representative
    example = keys[0]
    with h5py.File(path, 'r') as f:
        shape = f[example].shape
    n_samples = shape[0]
    return n_samples
    

parser = argparse.ArgumentParser(description="Simple argument parser example")
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--nb_epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--init', type=str, default='uniform', help='Initialization method: normal, uniform')
parser.add_argument('--nn', type=str, default='egnn', help='Neural network type: egnn, egnn_noe, egnn_lj, mlp')
parser.add_argument('--reflow', action='store_true', help='Use reflow data')
parser.add_argument('--prefix', type=str, default='flow_model', help='Prefix for output files')
parser.add_argument('--data_path', type=str, default='../lj.h5', help='Path to reflow data')
parser.add_argument('--ckpt_dir', type=str, default=None)  # optional override
parser.add_argument('--nparticles', type=int, default=16)
parser.add_argument('--kT', type=float, default=1.0)
parser.add_argument('--boxlength', type=float, default=0.0)
parser.add_argument('--dim', type=int, default=2)


args = parser.parse_args()
args_as_str = args_to_str(args)
seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.boxlength <= 0.0:
    args.boxlength = args.nparticles ** (1 / args.dim)

nparticles = args.nparticles
dim = args.dim
ljsystem= LJ(
  nparticles=nparticles,
  dim=dim,
  device='cuda',
  boxlength= args.boxlength,
  sigma=2 ** (-1 / 6),
  kT=args.kT,
  periodic=True
)


batch_size = args.batch_size
if args.reflow:
    # For reflow training we load the pre-generated samples directly
    # from a torch serialized file. This keeps the data in memory and
    # avoids the per-sample HDF5 reads that made the original
    # ``train.py --reflow`` path slower than ``train_reflow.py``.
    reflow = torch.load(args.data_path)
    x0 = reflow['x0']
    xt = reflow['xt']
    dataset = torch.utils.data.TensorDataset(x0, xt)
    nbsamples = len(dataset)
else:
    # Standard training uses the raw trajectory data stored in an HDF5
    # file and generates source samples on the fly.
    nbsamples = min(int(1e6), get_frame_count(args.data_path))
    rd_data = generate_source_data(ljsystem, n_samples=nbsamples)
    dataset = H5Dataset(
        args.data_path, dataset_name='trajectory', rd_data=rd_data, n_samples=nbsamples
    )

# Split train/val
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

lr = float(args.learning_rate)
train_model()
