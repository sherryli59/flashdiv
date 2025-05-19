import torch.nn as nn
import torch
from architectures import FlowNet, FlowMLP
from einops import rearrange, repeat
from pytorch_lightning import LightningModule

class VarDimMLP(LightningModule):
    def __init__(self, in_dim, out_dim,hidden_dim, n_layers):
        super().__init__()
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers-1)] + [nn.Linear(hidden_dim, out_dim)])
        self.activation = nn.ReLU()


    def forward(self, x):
        xt = x.clone()
        for layer in self.layers[:-1]:
            xt = self.activation(layer(xt))
        xt = self.layers[-1](xt)
        return xt

class FactorizedFlow(FlowNet):
    """
    we want to implement something that factors out the diagonal
    """
    def __init__(self, dim, nbparticles):
        super().__init__(dim)
        self.flowmlps = nn.ModuleList([
            VarDimMLP(
                in_dim = (dim * nbparticles) + 1,
                out_dim = dim,
                hidden_dim = 64,
                n_layersdim =2) for _ in range(nbparticles)
        ])
        self.mlps = nn.ModuleList([
            VarDimMLP(
                in_dim = dim,
                out_dim = dim,
                hidden_dim = 64
                n_layersdim =2) for _ in range(nbparticles)
        ])


    def forward(self, x, t):
        """
        x = [batch_size, npart, dim]
        """
        xt = torch.zeros_like(x)
        for k in range(self.nbparticles):
            x_perp = x.clone()
            # all other particles
            x_perp[:,k] = torch.zeros_like(x[:,k])
            x_perp = rearrange(x__perp, 'b part dim -> b (part dim)')
            x_perp = self.time_embedding(x_, t)
            x_perp = self.flowmlps[k](x_, t)

            # one particle
            x_ = x[:,k].clone()
            x_ = self.mlps[k](x_)

            # combine
            xt[:,k] = x_ * x_perp

        return xt


    @torch.no_grad()
    def divergence(self, x,t):
        """
        Divergence of the flow field
        """
        raise NotImplementedError("Divergence not implemented for FactorizedFlow")