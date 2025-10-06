import torch
import torch.nn as nn
from flashdiv.flows.flownet import FlowNet
from einops import rearrange, repeat, reduce

class EGNN_dynamicsPeriodic(FlowNet):
    def __init__(self, n_particles, n_dimension, hidden_nf=64, device='cpu',distribution=None,
            act_fn=torch.nn.SiLU(), n_layers=4, recurrent=True, attention=False, cutoff=None,max_neighbors=None,
                 condition_time=True, tanh=False, mode='egnn_dynamics', agg='sum', out_node_nf=None, boxlength=None, **kwargs):
        super().__init__()
        print('Initializing custom EGNN_dynamics')
        self.mode = mode

        # dirty hack to monitor the contribution of the lj potential.
        self.distribution = distribution

        self.out_node_nf = out_node_nf
        self.egnn = EGNN(in_node_nf=1, in_edge_nf=1, hidden_nf=hidden_nf, device=device, act_fn=act_fn, n_layers=n_layers, recurrent=recurrent, attention=attention, tanh=tanh, agg=agg, out_node_nf=self.out_node_nf)
        self.device = device
        self._n_particles = n_particles
        if max_neighbors is not None:
            self.max_nb_neighbors = max_neighbors
        else:
            self.max_nb_neighbors = n_particles
        self._n_dimension = n_dimension
        self.edges = self._create_edges()
        self._edges_dict = {}
        self.condition_time = condition_time
        self.cutoff = cutoff if cutoff is not None else 100.0  # Default cutoff distance
        self.boxlength = boxlength if boxlength is not None else 1000.0  # Default box length
        # Count function calls
        self.counter = 0

        self.pot_model = nn.Sequential(
            nn.Linear(1 + 1 + self.out_node_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1)
        )

        # self.max_nb_neighbors = 10  # P-1 neighbors for each particle
        self.pot_com_model = nn.Sequential(
            nn.Linear(1 + 1 + (self.max_nb_neighbors), hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1)
        )

    def _compute_common_terms(self, xs, t):
        """Shared preprocessing for forward/divergence style computations."""
        source_xs = xs.clone()
        B, P, D = xs.shape
        xs_neighbors = repeat(xs, 'b p d -> b p1 p d', p1=P)
        mask = torch.eye(P, device=self.device, dtype=bool).reshape(1, P, P, 1).expand(B, -1, -1, D)
        xs_neighbors = xs_neighbors[~mask].reshape(B, P, P - 1, D)

        particle_diffs = xs_neighbors - source_xs.unsqueeze(-2).expand(-1, -1, P - 1, -1)
        particle_diffs = particle_diffs % self.boxlength
        to_subtract = ((torch.abs(particle_diffs) > 0.5 * self.boxlength)
                       * torch.sign(particle_diffs) * self.boxlength)
        particle_diffs = particle_diffs - to_subtract
        del to_subtract

        xs_neighbors = source_xs.unsqueeze(-2).expand(-1, -1, P - 1, -1) + particle_diffs

        distances = (xs_neighbors - source_xs.unsqueeze(2)).norm(dim=-1)
        _, idx = distances.sort(dim=2)
        del distances

        idx = idx[:, :, :self.max_nb_neighbors]
        xs_neighbors = torch.gather(xs_neighbors, 2, idx.unsqueeze(-1).expand(-1, -1, -1, D))
        del idx

        com = xs_neighbors.mean(dim=2)
        source_xs_com = source_xs - com

        xs_centered = rearrange(xs_neighbors, 'b p1 p2 d -> (b p1) p2 d')
        xs_centered = xs_centered - xs_centered.mean(dim=1, keepdim=True)

        n_batch = xs_centered.shape[0]
        t_flat = t.reshape(-1, 1)

        edges = self.compute_edges(xs_centered, cutoff=self.cutoff)
        edges = [edges[0], edges[1]]

        x = xs_centered.reshape(-1, self._n_dimension).clone()
        h = torch.ones(n_batch, self.max_nb_neighbors, device=self.device)

        if self.condition_time:
            t_h = repeat(t_flat, 'b 1 -> (b p) 1', p=P)
            h = h * t_h
        h = h.reshape(-1, 1)

        edge_attr = torch.sum((x[edges[0]] - x[edges[1]])**2, dim=1, keepdim=True)
        h_final, x_final = self.egnn(h, x, edges, edge_attr=edge_attr)

        h_final = h_final.reshape(B, P, self.max_nb_neighbors, -1)
        vel = x_final.reshape(B, P, self.max_nb_neighbors, D)

        diffij = vel - source_xs_com.unsqueeze(2).expand(B, P, self.max_nb_neighbors, D)
        t_neighbors = t_flat.reshape(-1, 1, 1).expand(-1, P, self.max_nb_neighbors)

        return {
            'diffij': diffij,
            'h_final': h_final,
            'source_xs_com': source_xs_com,
            't_neighbors': t_neighbors,
            'B': B,
            'P': P,
            'D': D
        }

    def _create_edges(self):
        rows, cols = [], []
        for i in range(self._n_particles):
            for j in range(i + 1, self._n_particles):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def forward(self, xs, t):
        common = self._compute_common_terms(xs, t)
        diffij = common['diffij']
        h_final = common['h_final']
        B = common['B']
        P = common['P']
        n_neighbors = diffij.shape[2]

        rij = rearrange(diffij.norm(dim=-1), 'b p m -> (b p m) 1')
        t_pot = common['t_neighbors'].reshape(-1, 1)
        h_flat = h_final.reshape(-1, h_final.shape[-1])

        pot_input = torch.cat((rij, t_pot, h_flat), dim=-1)
        pot = self.pot_model(pot_input).reshape(B, P, n_neighbors, 1)

        vel = (diffij * pot).sum(dim=2)
        self.counter += 1
        return vel

    def forward_and_divergence(self, xs, t, differentiable=True):
        common = self._compute_common_terms(xs, t)
        diffij = common['diffij']
        h_final = common['h_final']
        B = common['B']
        P = common['P']
        n_neighbors = diffij.shape[2]

        t_pot = common['t_neighbors'].reshape(-1, 1)
        h_flat = h_final.reshape(-1, h_final.shape[-1])

        with torch.enable_grad():
            rij = rearrange(diffij.norm(dim=-1), 'b p m -> (b p m) 1').requires_grad_(True)
            pot_input = torch.cat((rij, t_pot, h_flat), dim=-1)
            pot = self.pot_model(pot_input).reshape(B, P, n_neighbors, 1)

            vel = (diffij * pot).sum(dim=2)

            dpotdr = torch.autograd.grad(pot.sum(), rij, create_graph=differentiable, retain_graph=differentiable)[0]
            dpotdr = dpotdr.reshape(B, P, n_neighbors, 1)
            rij_reshaped = rij.reshape(B, P, n_neighbors, 1)

            divergence_term = (
                pot + (diffij * dpotdr * diffij / (rij_reshaped + 1e-8))
            ).sum((-1, -2))

        divergence = -divergence_term.sum(-1)
        self.counter += 1
        return vel, divergence

    def divergence(self, xs, t, differentiable=True):
        common = self._compute_common_terms(xs, t)
        diffij = common['diffij']
        h_final = common['h_final']
        B = common['B']
        P = common['P']
        n_neighbors = diffij.shape[2]

        with torch.enable_grad():
            rij = rearrange(diffij.norm(dim=-1), 'b p m -> (b p m) 1').requires_grad_(True)
            t_pot = common['t_neighbors'].reshape(-1, 1)
            h_flat = h_final.reshape(-1, h_final.shape[-1])

            pot_input = torch.cat((rij, t_pot, h_flat), dim=-1)
            pot = self.pot_model(pot_input).reshape(B, P, n_neighbors, 1)

            dpotdr = torch.autograd.grad(pot.sum(), rij, create_graph=differentiable, retain_graph=differentiable)[0]
            dpotdr = dpotdr.reshape(B, P, n_neighbors, 1)
            rij_reshaped = rij.reshape(B, P, n_neighbors, 1)

            divergence_term = (
                pot + (diffij * dpotdr * diffij / (rij_reshaped + 1e-8))
            ).sum((-1, -2))

        divergence = -divergence_term.sum(-1)
        return divergence

    def compute_edges(self, x, cutoff):
        """
        x: Tensor of shape (B, P, D)
        cutoff: float
        Returns: [rows, cols] with shape (n_edges,), indices across the batch.
        """

        B, P, D = x.shape

        # Compute pairwise distances for the whole batch at once

        # need to do it in a periodic way
        pvec = x.unsqueeze(-3) - x.unsqueeze(-2)  # shape (B, P, P, D)
        pvec = pvec % self.boxlength  # wrap around the box
        to_subtract = ((torch.abs(pvec) > 0.5 * self.boxlength) * torch.sign(pvec) * self.boxlength)
        pvec = pvec - to_subtract  # right direction
        dist = torch.norm(pvec, dim=-1)  # shape (B, P, P)
        # dist = torch.cdist(x, x)  # shape (B, P, P)

        # Create a batch-wide diagonal mask to exclude self-loops
        device = x.device
        eye = torch.eye(P, dtype=torch.bool, device=device).unsqueeze(0)  # (1, P, P)

        # Create cutoff mask
        mask = (dist < cutoff) & (~eye)  # shape (B, P, P)
        # Get indices of edges where mask is True
        batch_idx, rows, cols = torch.nonzero(mask, as_tuple=True)  # shape (n_edges,)
        # Shift particle indices for each batch
        rows = rows + batch_idx * P
        cols = cols + batch_idx * P

        return [rows, cols]

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

class EGNN_dynamics(FlowNet):
    def __init__(self, n_particles, n_dimension, hidden_nf=64, device='cpu',
            act_fn=torch.nn.SiLU(), n_layers=4, recurrent=True, attention=False, cutoff=None,max_neighbors=None,
                 condition_time=True, tanh=False, mode='egnn_dynamics', agg='sum', out_node_nf=None):
        super().__init__()
        print('Initializing custom EGNN_dynamics')
        self.mode = mode
        self.out_node_nf = out_node_nf
        if mode == 'egnn_dynamics':
            self.egnn = EGNN(in_node_nf=1, in_edge_nf=1, hidden_nf=hidden_nf, device=device, act_fn=act_fn, n_layers=n_layers, recurrent=recurrent, attention=attention, tanh=tanh, agg=agg, out_node_nf=self.out_node_nf)
        else:
            raise ValueError(f"Mode {mode} is not supported. Use 'egnn_dynamics'.")
        self.device = device
        self._n_particles = n_particles
        if max_neighbors is not None:
            self.max_nb_neighbors = max_neighbors
        else:
            self.max_nb_neighbors = n_particles
        self._n_dimension = n_dimension
        self.edges = self._create_edges()
        self._edges_dict = {}
        self.condition_time = condition_time
        self.cutoff = cutoff if cutoff is not None else 100.0  # Default cutoff distance
        # Count function calls
        self.counter = 0

        self.pot_model = nn.Sequential(
            nn.Linear(1 + 1 + self.out_node_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1)
        )

        # self.max_nb_neighbors = 10  # P-1 neighbors for each particle
        self.pot_com_model = nn.Sequential(
            nn.Linear(1 + 1 + (self.max_nb_neighbors), hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1)
        )

    def _create_edges(self):
        rows, cols = [], []
        for i in range(self._n_particles):
            for j in range(i + 1, self._n_particles):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def forward(self, xs, t):

        # create subgraphs
        source_xs = xs.clone()
        B, P, D = xs.shape
        xs = repeat(xs, 'b p d -> b p1 p d', p1=P)
        mask = torch.eye(P, device=self.device, dtype=bool).reshape(1, xs.shape[1], xs.shape[1], 1).expand(B, -1, -1, D)
        xs = xs[~mask].reshape(B, P, P-1, D)

        distances = (xs - source_xs.unsqueeze(2)).norm(dim=-1)  # shape: (B, P, P-1)
        distances, idx = distances.sort(dim=2)
        # Select the P-1 closest particles for each particle


        idx = idx[:, :, :self.max_nb_neighbors]  # shape: (B, P, P-1)
        xs = torch.gather(xs, 2, idx.unsqueeze(-1).expand(-1, -1, -1, D))

        com = xs.mean(dim=2)
        source_xs_com = source_xs - com

        rcom = source_xs_com.norm(dim=-1, keepdim=True)  # (B,P,1)
        # print(xs.shape)
        xs = rearrange(xs, 'b p1 p2 d -> (b p1) p2 d') # we will pass this through the egnn
        xs = xs - xs.mean(dim=1, keepdim=True)  # remove mean

        n_batch = xs.shape[0]
        t = t.reshape(-1, 1)
        edges = self.compute_edges(xs, cutoff=self.cutoff)
        edges = [edges[0], edges[1]]
        x = xs.reshape(-1, self._n_dimension).clone()
        h = torch.ones(n_batch, self.max_nb_neighbors).to(self.device)

        if self.condition_time:
            t_ = repeat(t, 'b 1 -> (b p) 1', p=P)
            h = h * t_
        h = h.reshape(-1, 1)
        if self.mode == 'egnn_dynamics':
            edge_attr = torch.sum((x[edges[0]] - x[edges[1]])**2, dim=1, keepdim=True)
            h_final, x_final = self.egnn(h, x, edges, edge_attr=edge_attr)

            # only take xfinal here instead of diff
            vel = x_final

        h_final = h_final.reshape(B, P, self.max_nb_neighbors, -1) # reshape to match subgraph shape
        vel = vel.reshape(B, P, self.max_nb_neighbors, D) # should work

        diffij = source_xs_com.unsqueeze(2).expand(B, P, self.max_nb_neighbors, D) - vel
        rij = rearrange(
            diffij.norm(dim=-1),
            'b p p2 -> (b p p2) 1')
        t_ = t.reshape(-1,1,1).expand(-1, P,  self.max_nb_neighbors).reshape(-1,1) # (B P, P-1)should be same shape as rij

        pot = self.pot_model(
            torch.cat((
                rij,
                t_,
                h_final.reshape(-1, h_final.shape[-1])), dim=-1)).reshape(B, P, self.max_nb_neighbors, 1)

        # print(rcom.shape,t.reshape(-1, 1,1).expand(-1, P,  1).reshape(-1,1).shape,  h_final[:,:,:,-1].shape)
        com_pot = self.pot_com_model(
            torch.cat((
                rcom.reshape(-1, 1),
                t.reshape(-1, 1,1).expand(-1, P,  1).reshape(-1,1), #(B * P, 1)
                h_final[:,:,:,-1].reshape(B * P, -1),
            ), dim=-1)).reshape(B, P, 1)  # compute coercive potential

        vel = (diffij * pot).sum(dim=2) + (com_pot * source_xs_com) # sum over the P-1 particles
        self.counter += 1
        return vel

    def compute_edges(self, x, cutoff):
        """
        x: Tensor of shape (B, P, D)
        cutoff: float
        Returns: [rows, cols] with shape (n_edges,), indices across the batch.
        """

        B, P, D = x.shape

        # Compute pairwise distances for the whole batch at once
        pvec = x.unsqueeze(-3) - x.unsqueeze(-2)  # shape (B, P, P, D)
        dist = torch.norm(pvec, dim=-1)  # shape (B, P, P)
        # Create a batch-wide diagonal mask to exclude self-loops
        device = x.device
        eye = torch.eye(P, dtype=torch.bool, device=device).unsqueeze(0)  # (1, P, P)

        # Create cutoff mask
        mask = (dist < cutoff) & (~eye)  # shape (B, P, P)
        # Get indices of edges where mask is True
        batch_idx, rows, cols = torch.nonzero(mask, as_tuple=True)  # shape (n_edges,)
        # Shift particle indices for each batch
        rows = rows + batch_idx * P
        cols = cols + batch_idx * P

        return [rows, cols]

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

class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), recurrent=True, attention=False, clamp=False, norm_diff=True, tanh=False, coords_range=1, agg='sum'):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.agg_type = agg
        self.tanh = tanh
        edge_coords_nf = 1


        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = coords_range

        self.coord_mlp = nn.Sequential(*coord_mlp)
        self.clamp = clamp

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

        #if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)


    def edge_model(self, source, target, radial, edge_attr, edge_mask):

        #print("edge_model", radial, edge_attr)
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val

        if edge_mask is not None:
            out = out * edge_mask
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        #print("node_model", edge_attr)
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, radial, edge_feat, node_mask, edge_mask):
        #print("coord_model", coord_diff, radial, edge_feat)
        row, col = edge_index
        if self.tanh:
            trans = coord_diff * self.coord_mlp(edge_feat) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(edge_feat)
        #trans = torch.clamp(trans, min=-100, max=100)
        if edge_mask is not None:
            trans = trans * edge_mask

        if self.agg_type == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.agg_type == 'mean':
            if node_mask is not None:
                #raise Exception('This part must be debugged before use')
                agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
                M = unsorted_segment_sum(node_mask[col], row, num_segments=coord.size(0))
                agg = agg/(M-1)
            else:
                agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception("Wrong coordinates aggregation type")
        #print("update", coord, coord_diff,edge_feat, self.coord_mlp(edge_feat), self.coords_range, agg, self.tanh)
        coord = coord + agg
        return coord

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr, edge_mask)
        coord = self.coord_model(coord, edge_index, coord_diff, radial, edge_feat, node_mask, edge_mask)

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col  ], u, batch)  # GCN
        # print("h", h)
        if node_mask is not None:
            h = h * node_mask
            coord = coord * node_mask
        return h, coord, edge_attr

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        norm = torch.sqrt(radial + 1e-8)
        coord_diff = coord_diff/(norm + 1)

        return radial, coord_diff

def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)
