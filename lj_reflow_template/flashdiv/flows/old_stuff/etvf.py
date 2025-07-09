from flashdiv.flows.flow_net_torchdiffeq import FlowNet
from flashdiv.flows.egnn_et import EasyTrace_EGNN
from flashdiv.flows.eqtf_pair import EqTransformerFlow
import torch
import torch.nn as nn

class EasyTraceVelocityField(FlowNet):
    def __init__(self, n_particles, gnn_hidden_dim=64,tf_hidden_dim = 256, act_fn=torch.nn.SiLU(), n_layers=4, recurrent=True, attention=False, tanh=False, agg='sum', *, remove_com=True):
        super().__init__()
        self.n_particles = n_particles
        self.egnn = EasyTrace_EGNN(n_particles=n_particles, hidden_nf=gnn_hidden_dim,
                                   act_fn=act_fn, n_layers=n_layers, recurrent=recurrent,
                                   attention=attention, tanh=tanh, agg=agg, remove_com=remove_com)
        self.eqtf = EqTransformerFlow(n_particles=n_particles, hidden_nf=tf_hidden_dim)

    def forward(self, x, t):
        """
        Forward pass of the EasyTraceVelocityField model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, P, D) where B is the batch size,
                              P is the number of particles, and D is the dimension.
            t (torch.Tensor): Time tensor of shape (B,).

        Returns:
            torch.Tensor: Output tensor of shape (B, P, D) representing the velocity field.
        """
        P = x.shape[1]
        gnn_weights, xdiff = self.egnn.compute_weights(x, t, return_xdiff=True)
        attention_weights = self.eqtf.compute_weights(x, t)
        #remove diagnoal elemtns from attention_weights
        mask = ~torch.eye(P, dtype=torch.bool)
        attention_weights = attention_weights[:, mask].reshape(-1, P, P-1, 1)
        v  = (xdiff * (gnn_weights.unsqueeze(-1) * attention_weights)).sum(dim=2)          # (B , P , D)
        return v

