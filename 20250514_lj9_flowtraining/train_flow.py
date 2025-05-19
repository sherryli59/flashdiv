import torch
import torch.nn as nn
import numpy as np
import sys
from einops import rearrange, repeat, reduce
sys.path.append('.')

from core.flows.architectures import FlowNet,  VelocityBlock
from core.flows.trainer import FlowTrainer
from core.lj.utils import twod_rotation

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

sys.path.append("20250510_lj_samples")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# useful LJ Flow

class VelocityFlowLJ(FlowNet):
    def __init__(self, dim, hidden_dim, num_layers):
        super().__init__(dim)
        self.encoder = nn.Linear(self.dim+1, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, self.dim)
        self.layers = nn.ModuleList([VelocityBlock(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.time_embedding = lambda x, t: torch.cat([x, t], dim=1)

    def forward(self, x, t):
        xt = x.clone()
        x = rearrange(x, 'b part dim -> b (part dim)')
        xt = self.time_embedding(x, t)
        xt = self.encoder(xt)
        for layer in self.layers:
            xt = layer(xt)
        x = self.decoder(xt)
        return rearrange(x, 'b (part dim) -> b part dim', dim=2)

# load samples

temp = float(sys.argv[1])
nbparticles = 9
dim = 2
boxlength = 3.0

fname = f'lj9_2d_{temp}'
lj = torch.tensor(np.load(f'{fname}.npy'))
lj = (rearrange(lj, 'steps batch part dim -> (steps batch) part dim')) % boxlength

# enhance target data
target_data_new = twod_rotation(lj, nbrot=100) # 10^7 datapoints
target_data_new = target_data_new[torch.arange(target_data_new.size(0)).unsqueeze(-1), torch.argsort(target_data_new[:, :, 0], dim=1)] # sort by x


# prepare the random source data
rd_data = torch.randn_like(target_data_new).to(target_data_new) * 0.5
rd_data = rd_data[torch.arange(rd_data.size(0)).unsqueeze(-1), torch.argsort(rd_data[:, :, 0], dim=1)] # sort by x
rd_cm = repeat(reduce(rd_data, 'b p d -> b 1 d', 'mean'), 'b 1 d -> b p d', p=nbparticles)
rd_data = rd_data - rd_cm

dataset = torch.utils.data.TensorDataset(rd_data, target_data_new)
nb_data = len(dataset)

# split data into test and train
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size=256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# define trainer

## these hyperparameters are quite arbitrary for now... we take 10x vector size and take 4 layers;
# --> We're just trying to go as big as possible here.
# not that eventually the loss stops going down, but we'll see if we have major overfit by monitoring the val loss.

lr = float(sys.argv[2])
#lr = 1e-3 # I do wonder wether or not that's too agressive here...
velocitynet = VelocityFlowLJ(dim=int(nbparticles * dim), hidden_dim=int(6 * dim * nbparticles), num_layers=6)
velocitynet.to(device)
velocitytrainer = FlowTrainer(velocitynet, learning_rate=lr)

# train and save in right spot


nb_epochs = int(sys.argv[3])


trainer = pl.Trainer(max_epochs=nb_epochs,
                           accelerator="gpu" if torch.cuda.is_available() else "cpu",
                           default_root_dir=f'flow_model_{fname}_nb_data_{nb_data}_batch_size_{batch_size}_epochs_{nb_epochs}_lr_{lr}',
                           enable_progress_bar = False,
                           )

trainer.fit(velocitytrainer, train_loader, val_loader)
