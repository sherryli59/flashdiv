from flashdiv.flows.flownet import FlowNet
import torch.nn as nn
import torch
from einops import rearrange


class MLP(FlowNet):
    def __init__(self, dim, hidden_dim, num_layers, time_embed_dim: int | None = None):
        super().__init__()
        self.state_dim = dim  # flattened particle dimension
        self.hidden_dim = hidden_dim
        self.depth = num_layers
        self.time_embed_dim = time_embed_dim or hidden_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.SiLU(),
        )

        layers: list[nn.Module] = []
        in_dim = self.state_dim + self.time_embed_dim
        if self.depth <= 0:
            layers.append(nn.Linear(in_dim, self.state_dim))
        else:
            for _ in range(self.depth):
                layers.append(nn.Linear(in_dim, self.hidden_dim))
                layers.append(nn.SiLU())
                in_dim = self.hidden_dim
            layers.append(nn.Linear(in_dim, self.state_dim))
        self.net = nn.Sequential(*layers)

        self.time_embedding = self._concat_time_embedding

    def _concat_time_embedding(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_vec = t.reshape(-1, 1)
        t_emb = self.time_mlp(t_vec)
        return torch.cat([x, t_emb], dim=-1)

    def forward(self, x, t):
        batch, npart, dim = x.shape
        x_flat = rearrange(x, 'b part d -> b (part d)')
        xt = self.time_embedding(x_flat, t)
        out_flat = self.net(xt)
        return rearrange(out_flat, 'b (part d) -> b part d', part=npart, d=dim)
