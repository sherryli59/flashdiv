import torch
import torch.nn as nn
import numpy as np
import sys
from einops import rearrange, repeat, reduce

from flashdiv.flows.flow_net_torchdiffeq import FlowNet
from flashdiv.flows.eqtf_com import EqTransformerFlowSherryVariation
from flashdiv.flows.trainer import FlowTrainer
from flashdiv.flows.egnn_et import EasyTrace_EGNN
from flashdiv.flows.egnn import EGNN_dynamics
from flashdiv.flows.etvf import EasyTraceVelocityField
from flashdiv.flows.eqtf_pair import EqTransformerFlow
from flashdiv.flows.new_vf import EqTransformerFlow as NewEqTransformerFlow
from flashdiv.flows.egnn_new.egnn import EGNN_dynamics_Noe
from flashdiv.flows.egnn_new.egnn_var import EGNN_dynamics as EGNN_dynamics_var



from pytorch_lightning import Trainer, LightningModule
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import argparse

# parser details
parser = argparse.ArgumentParser(description="Simple argument parser example")
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--nb_epochs', type=int, default=60, help='Number of epochs')
parser.add_argument('--gnn_hidden_dim', type=int, default=32, help='hidden egnn layers')
parser.add_argument('--tf_hidden_dim', type=int, default=256, help='hidden eqtf layers')
parser.add_argument('--temp', type=float, default=1.0, help='temperature')
parser.add_argument('--nb_layers', type=int, default=4, help='nb layers of egnn network')
parser.add_argument('--nn', type=str, default='etvf', help='neural network architecture to use (default: etvf)')
parser.add_argument(
    '--resume_dir',
    type=str,
    default=None,
    help='Directory that contains a Lightning checkpoint (*.ckpt) to resume from'
)
parser.add_argument(
    '--ckpt_name',
    type=str,
    default='last.ckpt',
    help='Filename of the checkpoint inside --resume_dir (default: last.ckpt)'
)

args = parser.parse_args()

def args_to_str(args, ignore=("resume_dir", "ckpt_name")):
    """
    Turn the argparse.Namespace into a compact string for the run-folder,
    but *exclude* any keys listed in `ignore`.
    """
    parts = []
    for k, v in vars(args).items():
        if k not in ignore:
            parts.append(f"{k}_{v}")
    return "_".join(parts)

args_as_str = args_to_str(args)



# load samples

temp = float(args.temp)
nbparticles = 13
dim = 3
boxlength = None

# load the data and take 10^6 of them
# load the data and take 10^6 of them
fname = f'lj13_samples_noe_reflow'

data = torch.tensor(np.load(f'{fname}.npy')).clone().detach()
rd_data, target_data_new = data[0], data[1]
print(data.shape, rd_data.shape)



dataset = torch.utils.data.TensorDataset(rd_data, target_data_new)
nb_data = len(dataset)

# split data into test and train
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = int(args.batch_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

lr = float(args.learning_rate)


if args.nn == 'egnn':
    velocitynet = EGNN_dynamics(n_particles=nbparticles, n_dimension=dim, device=device, hidden_nf=int(args.gnn_hidden_dim),
        act_fn=torch.nn.SiLU(), n_layers=int(args.nb_layers), recurrent=True, tanh=True, attention=True, agg='sum')
elif args.nn == 'egnn_et':
    velocitynet = EasyTrace_EGNN(n_particles=nbparticles, hidden_nf=int(args.gnn_hidden_dim),
        act_fn=torch.nn.SiLU(), n_layers=int(args.nb_layers), recurrent=True, tanh=True, attention=True, agg='sum')
elif args.nn == "etvf":
    velocitynet = EasyTraceVelocityField(n_particles=nbparticles,gnn_hidden_dim = int(args.gnn_hidden_dim),
                                         tf_hidden_dim = int(args.tf_hidden_dim),act_fn=torch.nn.SiLU(), n_layers=int(args.nb_layers), 
                                         recurrent=True, tanh=True, attention=True, agg='sum')
elif args.nn == "eqtf_sherry":
    velocitynet = EqTransformerFlowSherryVariation(
        input_dim=3,
        embed_dim=int(args.tf_hidden_dim))
elif args.nn == "eqtf":
    velocitynet = EqTransformerFlow(
        n_particles=nbparticles,
        hidden_nf=int(args.tf_hidden_dim))
elif args.nn == "new_vf":
    velocitynet = NewEqTransformerFlow(
        n_particles=nbparticles,
        hidden_nf=int(args.tf_hidden_dim),
        gnn_hidden_nf=int(args.gnn_hidden_dim),
        )
elif args.nn == "egnn_noe":
    velocitynet = EGNN_dynamics_Noe(
        n_particles=nbparticles - 1,
        device=device,
        n_dimension=dim,
        hidden_nf=12,
        act_fn=torch.nn.SiLU(),
        n_layers=2,
        recurrent=True,
        tanh=True,
        attention=True,
        condition_time=True,
        # in_node_nf=1,  # 1 for time, 2 for position
        out_node_nf=12, # expressivity for potential
        )
elif args.nn == "egnn_var":
    velocitynet = EGNN_dynamics_var(
        n_particles=nbparticles,
        device=device,
        n_dimension=dim,
        hidden_nf=12,
        act_fn=torch.nn.SiLU(),
        n_layers=2,
        recurrent=True,
        tanh=True,
        attention=True,
        condition_time=True,
        # in_node_nf=1,  # 1 for time, 2 for position
        out_node_nf=12, # expressivity for potential
        )
    
    
#print number of parameters
print(f'Number of parameters in the model: {sum(p.numel() for p in velocitynet.parameters())}')

velocitynet.to(device)


ckpt_cb = ModelCheckpoint(
    dirpath=f"flow_model_{args_as_str}/checkpoints",
    filename="epoch={epoch}-step={step}",
    save_last=True,         # → writes .../checkpoints/last.ckpt
    save_top_k=1,           # keep best model too
    monitor="val_loss",
    mode="min"
)

# (optional) record the LR schedule in TensorBoard / CSV
lr_cb = LearningRateMonitor(logging_interval="epoch")

velocitytrainer = FlowTrainer(velocitynet, learning_rate=lr)


nb_epochs = int(args.nb_epochs)


trainer = Trainer(
    max_epochs=nb_epochs,
    #max_epochs = 1,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    default_root_dir=f'flow_model_{args_as_str}',
    enable_progress_bar = False,
    callbacks=[ckpt_cb, lr_cb],
    )

resume_ckpt = (
    str(Path(args.resume_dir) / args.ckpt_name)
    if args.resume_dir is not None else None
)

trainer.fit(velocitytrainer, train_loader, val_loader,ckpt_path=resume_ckpt)
