import torch
import torch.nn as nn

from flashdiv.flows.gcl import E_GCL_vel, E_GCL, GCL
from einops import rearrange, repeat, reduce
from torch.func import jvp
from flashdiv.flows.flow_net_torchdiffeq import FlowNet




class EGNN_dynamicsPeriodic(FlowNet):
    def __init__(self, n_particles, n_dimension, boxlength=None, hidden_nf=64, device='cpu',
            act_fn=torch.nn.SiLU(), n_layers=4, recurrent=True, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', agg='sum'):
        super().__init__()
        self.mode = mode
        if mode == 'egnn_dynamics':
            self.egnn = EGNN(in_node_nf=1, in_edge_nf=1, hidden_nf=hidden_nf, device=device, act_fn=act_fn, n_layers=n_layers, recurrent=recurrent, attention=attention, tanh=tanh, agg=agg)
        elif mode == 'gnn_dynamics':
            self.gnn = GNN(in_node_nf=1 + n_dimension, in_edge_nf=0, hidden_nf=hidden_nf, out_node_nf=n_dimension, device=device, act_fn=act_fn, n_layers=n_layers, recurrent=recurrent, attention=attention)
        self.device = device
        self._n_particles = n_particles
        self._n_dimension = n_dimension
        self.edges = self._create_edges()
        self._edges_dict = {}
        self.condition_time = condition_time
        self.boxlength = boxlength
        # Count function calls
        self.counter = 0

    def forward(self, xs, t):

        n_batch = xs.shape[0]
        edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles)
        edges = [edges[0], edges[1]]
        x = xs.reshape(n_batch*self._n_particles, self._n_dimension).clone()
        h = torch.ones(n_batch, self._n_particles).to(x.device)
        if self.condition_time:
            t = t.unsqueeze(-1)
            h = h*t.to(x.device)  

        h = h.reshape(n_batch*self._n_particles, 1)
        self.counter += 1
        if self.mode == 'egnn_dynamics':
            dx = x[edges[0]] - x[edges[1]]
            if self.boxlength is not None:
                dx = dx - torch.round(dx / self.boxlength) * self.boxlength
            edge_attr = (dx ** 2).sum(dim=1, keepdim=True)
            _, x_final = self.egnn(h, x, edges, edge_attr=edge_attr)
            vel = x_final - x
            vel = vel.view(n_batch, self._n_particles, self._n_dimension)
            vel = remove_mean(vel)
            return vel

        elif self.mode == 'gnn_dynamics':
            h = torch.cat([h, x], dim=1)
            vel = self.gnn(h, edges)
            return vel.view(n_batch,  self._n_particles, self._n_dimension)
        

    def _create_edges(self):
        rows, cols = [], []
        for i in range(self._n_particles):
            for j in range(i + 1, self._n_particles):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def _cast_edges2batch(self, edges, n_batch, n_nodes):
        if n_batch not in self._edges_dict:
            self._edges_dict = {}
            rows, cols = edges
            rows_total, cols_total = [], []
            for i in range(n_batch):
                rows_total.append(rows + i * n_nodes)
                cols_total.append(cols + i * n_nodes)
            rows_total = torch.cat(rows_total).to(self.device)
            cols_total = torch.cat(cols_total).to(self.device)

            self._edges_dict[n_batch] = [rows_total, cols_total]
        return self._edges_dict[n_batch]
    
    @torch.no_grad()
    def divergence(self, x,t, div_samples=int(1e2)):
        """
        hutchison trace estimator
        """
        x0 = repeat(x, 'b p d -> (b r) p d', r=div_samples)
        t = repeat(t, 'b -> (b r)', r=div_samples)
        v = torch.randn_like(x0)

        ouptut, jacvp = jvp(lambda x : self.forward(x, t),
        (x0,),
        (v,),
        )
        # print(jacvp.shape)

        tr = reduce(
            reduce(
                rearrange(
                    v * jacvp,
                    '(b r) p d -> b r p d',
                    r=div_samples),
                'b r p d -> b r',
                'sum'),
            'b r  -> b ',
            'mean')

        # takes care of residual
        torch.cuda.empty_cache()

        return tr


    
class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, recurrent=True, attention=False, norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, agg='sum'):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)/self.n_layers
        if agg == 'mean':
            self.coords_range_layer = self.coords_range_layer * 19
        #self.reg = reg
        ### Encoder
        #self.add_module("gcl_0", E_GCL(in_node_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=recurrent, attention=attention, norm_diff=norm_diff, tanh=tanh, coords_range=self.coords_range_layer, agg=agg))

        self.to(self.device)

    def forward(self, h, x, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, recurrent=True, attention=False, out_node_nf=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=recurrent, attention=attention))

        self.to(self.device)

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h

def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x


def remove_mean_with_mask(x, node_mask):
    assert (x * (1 - node_mask)).abs().sum().item() < 1e-8
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x