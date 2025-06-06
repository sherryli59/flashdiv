import torch
import torch.nn as nn
from flashdiv.flows.architectures import FlowNet
from einops import rearrange, repeat, reduce
import torch.nn.functional as F
from torch_scatter import scatter_softmax
from torch_scatter import scatter

class EqTransformerFlowLJ(FlowNet):
    def __init__(self, input_dim, embed_dim=128, key_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.key_dim = key_dim

        # we have multipe encoders to avoid the symmetry issue.
        self.encoder =  nn.Sequential(
            nn.Linear(1 + 1, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )
        
        self.scaler = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
)

        self.edges = None
        self._edges_dict = {}
    def forward_old(self, x, t):
        """
        This is a rot and perm equivariant flow field, that uses the "softmax attention mechanism" on featured radial distances.
        In the Noe paper they compose multiple similar layers, but our bane, or advantage, is that we want only one
        for quick trace computations.

        Not that we can always put them in parallel. to have potentially more expressivity.

        Inputs
        x :  (batch_size, nbpart, dim) --> This should scale to nbparticles in nd
        t :  (batch_size)

        Outputs
        v :  (batch_size, nbpart, dim)
        """



        n_batch = x.shape[0]
        n_particles = x.shape[1]

        # steal MW's trick --> this assigns self.eges the first time.
        # Doesn't work for batches bcs batch size might very.
        if self.edges is None:
            self.edges = self._create_edges(n_particles)
        batches, edgesi, edgesj = self._cast_edges2batch(self.edges, n_batch, n_particles)
        edges = [edgesi.to(x.device), edgesj.to(x.device)]
        batches = batches.to(x.device)

        flat_coord = rearrange(x, 'b p d -> (b p) d')
        # compute pairwaise distances and direction information
        radial, coord_diff = self.coord2radial(edges, flat_coord) # one big matrix of size ((b p p) 1)
        flat_t = repeat(t, 'b -> (b p ) 1', p=(batches==0).sum()) # no cross terms


        flat_features = torch.cat((radial, flat_t), dim=-1)

        # pass features throught the encoder
        flat_features = self.encoder(flat_features) #((b p p ) 1)
        import time

        # now we have to reshape to do a soft max operation.
        sm = torch.zeros(
            (n_batch, n_particles, n_particles,1),
            device=x.device,
            dtype=flat_features.dtype
        )
        diffs = torch.zeros(
            (n_batch, n_particles, n_particles, x.shape[-1]),
            device=x.device,
            dtype=flat_features.dtype
        )
        #scales = torch.zeros_like(sm,device=x.device, dtype=flat_features.dtype)
        # sm2 = torch.zeros_like(sm)
        # diffs2 = torch.zeros_like(diffs)
        # start = time.perf_counter()
        # # do both coord_diffs and radial
        # # this operation I think is costly, but we don't really have a choice... Or at least I don't see ow to do it now.
        # for b,i,j,f,c in zip(batches, edges[0], edges[1], flat_features, coord_diff):
        #     sm2[b, i%n_particles,j%n_particles] = f
        #     diffs2[b, i%n_particles,j%n_particles] = c
        # end = time.perf_counter()
        # print(f"Elapsed time: {end - start:.6f} seconds")
        #start = time.perf_counter()
        i_mod = edges[0] % n_particles
        j_mod = edges[1] % n_particles

        sm[batches, i_mod, j_mod] = flat_features
        diffs[batches, i_mod, j_mod] = coord_diff

        # end = time.perf_counter()
        # print(f"Elapsed time: {end - start:.6f} seconds")
        # # check if the two methods are equal
        # assert torch.allclose(sm, sm2), "The two methods are not equal!"
        # assert torch.allclose(diffs, diffs2), "The two methods are not equal!"
        #softmax
        sm = F.softmax(sm, dim=-1)
        
        #multiply and sum
        out = reduce(
            sm * diffs, # (b, p, p, d),
            'b p1 p2 d -> b p1 d',
            'sum')


        return out
    def forward(self, x, t):
        """
        This is a rot and perm equivariant flow field, that uses the "softmax attention mechanism" on featured radial distances.
        In the Noe paper they compose multiple similar layers, but our bane, or advantage, is that we want only one
        for quick trace computations.

        Not that we can always put them in parallel. to have potentially more expressivity.

        Inputs
        x :  (batch_size, nbpart, dim) --> This should scale to nbparticles in nd
        t :  (batch_size)

        Outputs
        v :  (batch_size, nbpart, dim)
        """



        n_batch = x.shape[0]
        n_particles = x.shape[1]

        # steal MW's trick --> this assigns self.eges the first time.
        # Doesn't work for batches bcs batch size might very.
        if self.edges is None:
            self.edges = self._create_edges(n_particles)
        batches, edgesi, edgesj = self._cast_edges2batch(self.edges, n_batch, n_particles)
        edges = [edgesi.to(x.device), edgesj.to(x.device)]
        batches = batches.to(x.device)

        flat_coord = rearrange(x, 'b p d -> (b p) d')
        # compute pairwaise distances and direction information
        radial, coord_diff = self.coord2radial(edges, flat_coord) # one big matrix of size ((b p p) 1)
        flat_t = repeat(t, 'b -> (b p ) 1', p=(batches==0).sum()) # no cross terms


        flat_features = torch.cat((radial, flat_t), dim=-1)
        scale = self.scaler(flat_features)

        # pass features throught the encoder
        flat_features = self.encoder(flat_features) #((b p p ) 1)

        # # now we have to reshape to do a soft max operation.
        # sm = torch.zeros(
        #     (n_batch, n_particles, n_particles,1),
        #     device=x.device,
        #     dtype=flat_features.dtype
        # )
        # scale_tensor = torch.zeros_like(sm,device=x.device, dtype=flat_features.dtype)

        # diffs = torch.zeros(
        #     (n_batch, n_particles, n_particles, x.shape[-1]),
        #     device=x.device,
        #     dtype=flat_features.dtype
        # )
        i_mod = edges[0] % n_particles
        j_mod = edges[1] % n_particles

        # index = batches * (n_particles ** 2) + i_mod * n_particles + j_mod  # [E]
        # numel = n_batch * n_particles * n_particles

        # # Scatter flat_features into sm
        # flat_sm = scatter(flat_features, index, dim=0, dim_size=numel)  # [numel, F]
        # sm = flat_sm.view(n_batch, n_particles, n_particles, -1)        # [B, P, P, F]

        # # Scatter coord_diff into diffs
        # flat_diffs = scatter(coord_diff, index, dim=0, dim_size=numel)  # [numel, D]
        # diffs = flat_diffs.view(n_batch, n_particles, n_particles, -1)  # [B, P, P, D]

        # # Scatter scale into scale_tensor
        # flat_scale = scatter(scale, index, dim=0, dim_size=numel)       # [numel, 1]
        # scale_tensor = flat_scale.view(n_batch, n_particles, n_particles, 1)  # [B, P, P, 1]
        sm = []
        diffs = []
        scale_tensor = []

        for b in range(n_batch):
            idx_b = (batches == b)
            i_b = i_mod[idx_b]
            j_b = j_mod[idx_b]
            flat_idx = i_b * n_particles + j_b  # flatten i,j into a single dim

            # Gather per-batch tensors
            sm_b = torch.zeros((n_particles * n_particles, flat_features.shape[-1]), device=flat_features.device)
            diffs_b = torch.zeros((n_particles * n_particles, coord_diff.shape[-1]), device=coord_diff.device)
            scale_b = torch.zeros((n_particles * n_particles, 1), device=scale.device)

            # Use index_add_ (out-of-place update)
            sm_b = sm_b.index_add(0, flat_idx, flat_features[idx_b])
            diffs_b = diffs_b.index_add(0, flat_idx, coord_diff[idx_b])
            scale_b = scale_b.index_add(0, flat_idx, scale[idx_b])

            # Reshape to [P, P, ...]
            sm.append(sm_b.view(n_particles, n_particles, -1))
            diffs.append(diffs_b.view(n_particles, n_particles, -1))
            scale_tensor.append(scale_b.view(n_particles, n_particles, 1))

        # Stack results into [B, P, P, ...]
        sm = torch.stack(sm, dim=0)
        diffs = torch.stack(diffs, dim=0)
        scale_tensor = torch.stack(scale_tensor, dim=0)
        #softmax
        sm = sm.squeeze(-1)                  # shape: [B, N, N]
        sm = F.softmax(sm, dim=-1)          # softmax over j-neighbors
        sm = sm.unsqueeze(-1) 
        sm = sm * scale_tensor
        #multiply and sum
        out = reduce(
            sm * diffs, # (b, p, p, d),
            'b p1 p2 d -> b p1 d',
            'sum')
        return out

    def forward_new(self, x, t ):
        """
        Args:
            x: Tensor of shape [B, N_max, D] – batched particle coordinates
            t: Tensor of shape [B] – time scalar per graph
            edge_index: Tensor of shape [2, E_total] – global edges across batch
            n_particles_b: list of length B with number of particles per graph
        Returns:
            v_out_padded: Tensor of shape [B, N_max, D] – equivariant vector field
        """

        device = x.device
        n_particles = x.shape[1]
        edge_index = self._create_batched_edges(n_particles, x.shape[0], device)
        edge_index = [e.to(device) for e in edge_index]
        n_particles_b = [x.shape[1]] * x.shape[0]
        # Flatten node positions: (B * N_max, D)
        x_flat = rearrange(x, 'b n d -> (b n) d')

        # Create batch index for each node
        batch = torch.cat([
            torch.full((n,), b, dtype=torch.long, device=device)
            for b, n in enumerate(n_particles_b)
        ])

        # Compute pairwise vector and squared distance
        xi = x_flat[edge_index[0]]
        xj = x_flat[edge_index[1]]
        coord_diff = xi - xj                      # shape: [E, D]
        radial = (coord_diff ** 2).sum(dim=-1, keepdim=True)  # [E, 1]

        # Broadcast time to edges based on source node
        t_flat = t[batch][edge_index[0]].unsqueeze(-1)  # [E, 1]

        # Combine invariant features: [r², t]
        features = torch.cat([radial, t_flat], dim=-1)  # [E, 2]

        # Compute attention logits and weights (can be signed)
        logits = self.encoder(features).squeeze(-1)     # [E]
        attn_weights = scatter_softmax(logits, edge_index[0], dim=0)  # [E]

        # Compute scalar magnitude for coord_diff (must be positive)
        scale = self.scaler(features).squeeze(-1)       # [E]
        scaled_diff = scale.unsqueeze(-1) * coord_diff  # [E, D]

        # Apply attention
        messages = attn_weights.unsqueeze(-1) * scaled_diff  # [E, D]
        # Aggregate messages to source nodes
        v_out = torch.zeros_like(x_flat)               # [N_total, D]
        v_out.index_add_(0, edge_index[0], messages)

        # Unflatten output to [B, N_max, D]
        per_graph_outputs = torch.split(v_out, n_particles_b, dim=0)
        v_out_padded = torch.nn.utils.rnn.pad_sequence(per_graph_outputs, batch_first=True)

        return v_out_padded


    # This is a standard trick to have more parrallelizable computations between independent batches.
    def _create_edges(self, n_particles):
        rows, cols = [], []
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def _create_batched_edges(self, n_particles, batch_size, device):
        base_edge_index = self._create_edges(n_particles)  # list of two 1D tensors
        row, col = base_edge_index
        edge_list = []

        for b in range(batch_size):
            offset = b * n_particles
            edge = torch.stack([row + offset, col + offset], dim=0)  # [2, E]
            edge_list.append(edge)

        return torch.cat(edge_list, dim=1).to(device)
    
    def _cast_edges2batch(self, edges, n_batch, n_nodes):
        if n_batch not in self._edges_dict:
            self._edges_dict = {}
            rows, cols = edges
            rows_total, cols_total = [], []
            for i in range(n_batch):
                rows_total.append(rows + i * n_nodes)
                cols_total.append(cols + i * n_nodes)
            rows_total = torch.cat(rows_total)
            cols_total = torch.cat(cols_total)
            batches = rows_total // (n_nodes)

            self._edges_dict[n_batch] = [batches, rows_total, cols_total]
        return self._edges_dict[n_batch]

    # returns the dij, (xi-xj)
    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)

        norm = torch.sqrt(radial + 1e-8)
        return radial, coord_diff

    @torch.no_grad()
    def fast_divergence(self, x, t):
        """
        Inputs
        x :  (batch_size, nbpart, dim) --> weird but we want to try it in 2d first
        t :  (batch_size, 1)

        Outputs
        v :  (batch_size)
        """
        raise NotImplementedError("Equivariant divergence not implemented yet")
