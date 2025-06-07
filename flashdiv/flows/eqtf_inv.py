import torch
import torch.nn as nn
from flashdiv.flows.flow_net import FlowNet
from einops import rearrange, repeat, reduce
import torch.nn.functional as F

"""
Same architecture as before, but the features derive from the 1 / (dij - d_min)
"""

class EqTransformerFlowLJ(FlowNet):
    def __init__(self, input_dim, embed_dim=128, dmin=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.dmin = dmin  # minimum LJ potential distance

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

    def forward_vmap(self, x, t):
        """
        This is a rot and perm equivariant flow field, that uses the “softmax attention mechanism” on featured radial distances.
        In the Noe paper they compose multiple similar layers, but our bane, or advantage, is that we want only one
        for quick tract computations.
        Not that we can always put them in parallel. to have potentially more expressivity.
        Inputs
        x :  (batch_size, nbpart, dim) --> This should scale to nbparticles in nd
        t :  (batch_size)
        Outputs
        v :  (batch_size, nbpart, dim)
        """
        n_batch = x.shape[0]
        n_particles = x.shape[1]
        # steal MW’s trick --> this assigns self.eges the first time.
        # Doesn’t work for batches bcs batch size might very.
        if self.edges is None:
            self.edges = self._create_edges(n_particles)
        batches,edgesi,edgesj = self._cast_edges2batch(self.edges, n_batch, n_particles)
        edges = [edgesi.to(x.device), edgesj.to(x.device)]
        batches = batches.to(x.device)
        flat_coord = rearrange(x, 'b p d -> (b p) d')
        # compute pairwaise distances and direction information
        radial, coord_diff = self.coord2radial(edges, flat_coord) # one big matrix of size ((b p p) 1)
        sq_radial = radial ** 0.5
        sq_radial_shift = sq_radial - self.dmin
        feat_radial  = 1 / sq_radial_shift

        flat_t = repeat(t, 'b -> (b p ) 1', p=(batches==0).sum()) # no cross terms
        flat_features = torch.cat((feat_radial, flat_t), dim=-1)
        scale = self.scaler(flat_features) #((b p p ) 1)
        # pass features throught the encoder
        flat_features = self.encoder(flat_features) #((b p p ) 1)
        i_mod = edges[0] % n_particles
        j_mod = edges[1] % n_particles
        sm = []
        diffs = []
        scales = []
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
            scales.append(scale_b.view(n_particles, n_particles, 1))
        # Stack results into [B, P, P, ...]
        sm = torch.stack(sm, dim=0)
        diffs = torch.stack(diffs, dim=0)
        scales = torch.stack(scales, dim=0)
        #softmax
        sm = F.softmax(sm, dim=-2)
        sm = sm * scales
        #multiply and sum
        out = reduce(
            sm * diffs, # (b, p, p, d),
            'b p1 p2 d -> b p1 d',
            'sum')
        return out

    def forward(self, x, t):
        """
        This is a rot and perm equivariant flow field, that uses the “softmax attention mechanism” on featured radial distances.
        In the Noe paper they compose multiple similar layers, but our bane, or advantage, is that we want only one
        for quick tract computations.
        Not that we can always put them in parallel. to have potentially more expressivity.
        Inputs
        x :  (batch_size, nbpart, dim) --> This should scale to nbparticles in nd
        t :  (batch_size)
        Outputs
        v :  (batch_size, nbpart, dim)
        """
        n_batch = x.shape[0]
        n_particles = x.shape[1]
        # steal MW’s trick --> this assigns self.eges the first time.
        # Doesn’t work for batches bcs batch size might very.
        if self.edges is None:
            self.edges = self._create_edges(n_particles)
        batches,edgesi,edgesj = self._cast_edges2batch(self.edges, n_batch, n_particles)
        edges = [edgesi.to(x.device), edgesj.to(x.device)]
        batches = batches.to(x.device)
        flat_coord = rearrange(x, 'b p d -> (b p) d')
        # compute pairwaise distances and direction information
        radial, coord_diff = self.coord2radial(edges, flat_coord) # one big matrix of size ((b p p) 1)
        sq_radial = radial ** 0.5
        sq_radial_shift = sq_radial - self.dmin
        feat_radial  = 1 / sq_radial_shift

        flat_t = repeat(t, 'b -> (b p ) 1', p=(batches==0).sum()) # no cross terms
        flat_features = torch.cat((feat_radial, flat_t), dim=-1)
        scale = self.scaler(flat_features) #((b p p ) 1)
        # pass features throught the encoder
        flat_features = self.encoder(flat_features) #((b p p ) 1)
        # now we have to reshape to do a soft max operation.

        # these might have an extra particle, but it can be a problem for later.
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

        scales = torch.zeros_like(sm, device=x.device, dtype=flat_features.dtype)
        i_mod = edges[0] % n_particles
        j_mod = edges[1] % n_particles
        sm[batches, i_mod, j_mod] = flat_features
        diffs[batches, i_mod, j_mod] = coord_diff
        scales[batches, i_mod, j_mod] = scale
        #softmax
        sm = F.softmax(sm, dim=-2)
        sm = sm * scales
        #multiply and sum
        out = reduce(
            sm * diffs, # (b, p, p, d),
            'b p1 p2 d -> b p1 d',
            'sum')
        return out
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


    def divergence(self, x, t):
        """
        Inputs
        x :  (batch_size, nbpart, dim) --> weird but we want to try it in 2d first
        t :  (batch_size, 1)
        Outputs
        v :  (batch_size)
        """

        # start with essential compute
        n_batch = x.shape[0]
        n_particles = x.shape[1]
        # steal MW’s trick --> this assigns self.eges the first time.
        # Doesn’t work for batches bcs batch size might very.
        if self.edges is None:
            self.edges = self._create_edges(n_particles)
        batches,edgesi,edgesj = self._cast_edges2batch(self.edges, n_batch, n_particles)
        edges = [edgesi.to(x.device), edgesj.to(x.device)]
        batches = batches.to(x.device)
        flat_coord = rearrange(x, 'b p d -> (b p) d')
        # compute pairwaise distances and direction information
        radial, coord_diff = self.coord2radial(edges, flat_coord) # one big matrix of size ((b p p) 1)
        sq_radial = radial ** 0.5
        sq_radial_shift = sq_radial - self.dmin
        feat_radial  = 1 / sq_radial_shift


        # requires grad on radial
        feat_radial.requires_grad_(True)

        flat_t = repeat(t, 'b -> (b p ) 1', p=(batches==0).sum()) # no cross terms
        flat_features = torch.cat((feat_radial , flat_t), dim=-1)
        scale = self.scaler(flat_features) #((b p p ) 1)

        # compute gradient of scale.sum() w.r.t. feat_radial
        (dscale,) = torch.autograd.grad(
            outputs=scale.sum(),
            inputs=feat_radial,
            create_graph=True
        )


        # pass features throught the encoder
        flat_features = self.encoder(flat_features) #((b p p ) 1)

        # compute gradient of flat_features.sum() w.r.t. feat_radial
        (dfeat,) = torch.autograd.grad(
            outputs=flat_features.sum(),
            inputs=feat_radial,
            create_graph=True
            )


        # now we have to reshape to do a soft max operation.

        # these might have an extra particle, but it can be a problem for later.
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
        sq_radials = torch.zeros_like(sm, device=x.device, dtype=flat_features.dtype)
        scales = torch.zeros_like(sm, device=x.device, dtype=flat_features.dtype)
        dscales = torch.zeros_like(sm, device=x.device, dtype=dscale.dtype)
        dfeatures = torch.zeros_like(sm, device=x.device, dtype=dfeat.dtype)


        # fill in alll the arrays needed
        i_mod = edges[0] % n_particles
        j_mod = edges[1] % n_particles
        sm[batches, i_mod, j_mod] = flat_features
        diffs[batches, i_mod, j_mod] = coord_diff
        scales[batches, i_mod, j_mod] = scale
        dscales[batches, i_mod, j_mod] = dscale
        dfeatures[batches, i_mod, j_mod] = dfeat
        sq_radials[batches, i_mod, j_mod] = sq_radial



        #softmax
        sm = F.softmax(sm, dim=-2)

        # divergence vector to be filled
        divergence = torch.zeros(
            (n_batch, n_particles, x.shape[-1]),
            device=x.device,
            dtype=flat_features.dtype,
            requires_grad=False
        )

        #linear term

        divergence += reduce(
            sm * scales,  # (b, p, p, d),
            'b p1 p2 d -> b p1 d',
            'sum'
        )


        #scales term
        divergence += reduce(
            diffs *  sm * dscales * (-1) / (sq_radials - self.dmin) ** 2 * (diffs / sq_radials).nan_to_num(0.0),
            'b p1 p2 d -> b p1 d',
            'sum'
        )

        #sm term

        # non cross terms
        sm_ = repeat(
            (- sm * dfeatures * (-1) / (sq_radials - self.dmin) ** 2 * (diffs / sq_radials).nan_to_num(0.0)).sum(-2) ,
            'b p1 d -> b p1 p2 d',
            p2=n_particles
        )

        divergence += reduce(
            (sm_ * sm * scales * diffs),
            'b p1 p2 d -> b p1 d',
            'sum'
            )

        #cross terms
        divergence += reduce(
            (diffs * sm * scales * dfeatures * (-1) / (sq_radials - self.dmin) ** 2 * (diffs / sq_radials).nan_to_num(0.0)),
            'b p1 p2 d -> b p1 d',
            'sum'
            )

        # final sum
        divergence = reduce(
            divergence,
            'b p d -> b',
            'sum'
        )

        return divergence
