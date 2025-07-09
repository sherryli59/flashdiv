from flashdiv.flows.flow_net_torchdiffeq import FlowNet
import torch.nn as nn
import torch
from einops import rearrange

class MLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, xt):
        x = self.fc1(xt)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

class MLP(FlowNet):
    def __init__(self, dim, hidden_dim, num_layers):
        super().__init__()
        self.dim = dim
        self.encoder = nn.Linear(self.dim+1, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, self.dim)
        self.layers = nn.ModuleList([MLPBlock(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.time_embedding = lambda x, t: torch.cat([x, t], dim=1)

    def forward(self, x, t):
        npart = x.shape[-2]
        dim = x.shape[-1]
        xt = x.clone()
        x = rearrange(x, 'b part dim -> b (part dim)')
        xt = self.time_embedding(x, t.view(-1, 1))
        xt = self.encoder(xt)
        for layer in self.layers:
            xt = layer(xt)
        x = self.decoder(xt)
        return rearrange(x, 'b (part dim) -> b part dim', dim=dim)
