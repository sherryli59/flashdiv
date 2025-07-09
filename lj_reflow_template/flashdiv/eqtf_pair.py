import torch
import torch.nn as nn
#from flashdiv.flows.flow_net import FlowNet
from flashdiv.flows.flow_net_torchdiffeq import FlowNet
from einops import rearrange, repeat, reduce
import torch.nn.functional as F

class EqTransformerFlow(FlowNet):
    def __init__(self, n_particles, hidden_nf=128):
        super().__init__()
        self.n_particles = n_particles
        self.edges = self._create_edges(n_particles)
        self.hidden_nf = hidden_nf

        # we have multipe encoders to avoid the symmetry issue.
        self.encoder =  nn.Sequential(
            nn.Linear(1 + 1, self.hidden_nf),
            nn.ReLU(),
            nn.Linear(self.hidden_nf, 1)
        )

        self.scaler = nn.Sequential(
            nn.Linear(4, hidden_nf),
            nn.ReLU(),
            nn.Linear(hidden_nf, 1)
        )

        self.edges = None
        self._edges_dict = {}

    def compute_weights(self, x, t, return_grad=False):
        B, P, D = x.shape
        dev     = x.device
        if t.ndim == 0:                            # () → (B,)
            t = t.expand(B)
        elif t.numel() == 1 and B != 1:            # (1,) → (B,)
            t = t.expand(B)
        elif t.shape[0] != B:                      # mismatched batch
            raise ValueError(f"`t` length {t.shape[0]} ≠ batch size {B}")
        # ------------------------------------------------------------------ #
        # 1)  Pair bookkeeping                 #
        # ------------------------------------------------------------------ #
        if self.edges is None:
            self.edges = self._create_edges(P)

        batches, ei, ej = self._cast_edges2batch(self.edges, B, P)
        edges   = [ei.to(dev), ej.to(dev)]
        batches = batches.to(dev)
        flat_x  = rearrange(x, 'b p d -> (b p) d')
        radial_flat, diff_flat = self.coord2radial(edges, flat_x)        # (B·P·P,1)
        flat_t = t.repeat_interleave(radial_flat.shape[0] // t.shape[0]).unsqueeze(1)
        pair_feat  = torch.cat((radial_flat, flat_t), dim=-1)            # (B·P·P,2)
        logits     = self.encoder(pair_feat)                             # (B·P·P,1)

        # pair-attention tensor w_{kj}
        W = torch.zeros(B, P, P, 1, device=dev)
        W[batches, ei % P, ej % P] = logits
        W = F.softmax(W, dim=-2)          # soft-max over j
        W = W.unsqueeze(dim=1)
        # ------------------------------------------------------------------ #
        # 2)  Dense pair distance tensor  R(i,j)  →  (B,P,P,1)               #
        #     (your coord2radial gave (B,P,P-1,1); we fill the diagonal)     #
        # ------------------------------------------------------------------ #
        R = torch.zeros((B,P,P,1)).to(W.device)     # (B,P,P,1)
        R[batches, ei % P, ej % P,:] = radial_flat
        # ------------------------------------------------------------------ #
        # 3)  Build triplet features, BROADCAST to (B,P,P,P,1) everywhere    #
        # ------------------------------------------------------------------ #

        r_ij = R.unsqueeze(2).expand(-1, -1, P, -1, -1)   # (B,P,P,P,1)
        r_jk = R.unsqueeze(1).expand(-1, P, -1, -1, -1)   # (B,P,P,P,1)
        
        xi = x.unsqueeze(2).unsqueeze(3)        # (B, P,1,1,D)
        xj = x.unsqueeze(1).unsqueeze(3)        # (B,1,P,1,D)
        xk = x.unsqueeze(1).unsqueeze(2)        # (B,1,1,P,D)
        v1 = xi - xj                            # x_i − x_j   (B,P,P,1,D)
        v2 = xk - xj                            # x_k − x_j   (B,1,P,P,D)

        v1 = v1.expand(-1, -1, -1, P, -1)       # (B,P,P,P,D)
        v2 = v2.expand(-1, P, -1, -1, -1)       # (B,P,P,P,D)

        # ---------- (b)  cosine(∠ijk) = (v1·v2)/(|v1||v2|) -----------------
        dot   = (v1 * v2).sum(dim=-1, keepdim=True)          # (B,P,P,P,1)
        norm1 = v1.norm(dim=-1, keepdim=True) + 1e-8
        norm2 = v2.norm(dim=-1, keepdim=True) + 1e-8
        cosine = dot / (norm1 * norm2)                       # (B,P,P,P,1)


        # time broadcast
        t4 = t.view(B, 1, 1, 1, 1).expand(-1, P, P, P, 1)

        # ---------- (c)  final triplet feature tensor ----------------------
        triplet_feat = torch.cat((r_ij, r_jk, cosine, t4), dim=-1)
    
        # three-body scale  s_{ijk}
        s = self.scaler(triplet_feat)                            # (B,P,P,P,1)
        # if return_grad:
        #     # compute gradient of scale.sum() w.r.t. radial
        #     (dscale1,) = torch.autograd.grad(
        #         outputs=triplet_feat.sum(dim=(0,1,2)),
        #         inputs=r_ij,
        #         grad_outputs=torch.ones_like(triplet_feat.sum(dim=(0,1,2))),
        #         create_graph=True
        #     )
        #     s_iji = s.diagonal(offset=0, dim1=1, dim2=3)
        #     (dscale2,) = torch.autograd.grad(
        #         outputs=s_iji.sum(),
        #         inputs=r_jk,
        #         grad_outputs=torch.ones_like(s_iji.sum(dim=(0,1))),
        #         create_graph=True
        #     )
        # W : (B, i, j, 1)          -- already softmax-normalised
        # S_{ij} = Σ_k W_{kj} · s_{ijk}
        S = torch.einsum('bikjc, bikjc -> bijc', W, s)    # (B, i, j, 1)
        return S
        

    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Three-body rotational & permutation-equivariant flow.

        x : (B, P, D)   particle coordinates
        t : (B,)        times   (broadcasted inside)
        returns
        v : (B, P, D)   velocity field
        """
        B, P, D = x.shape
        W = self.compute_weights(x, t)  # (B,P,P,1)

        # ------------------------------------------------------------------ #
        # 4)  Displacements  (x_i - x_j)  →  (B,P,P,P,D)                     #
        # ------------------------------------------------------------------ #
        diff_ij = x.unsqueeze(2) - x.unsqueeze(1)        # (B,P,P,D)
        #diff    = diff_ij.unsqueeze(2).expand(-1, P, P, P, -1)

        # ------------------------------------------------------------------ #
        # 5)  Contract:   Σ_{k,j}  w_{kj} · s_{ijk} · (x_i - x_j)            #
        # ------------------------------------------------------------------ #
        v = (W * diff_ij).sum(dim=2)
        return v


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
    
    def divergence_exact(self,x,t, create_graph=False):
        shape = x.shape

        def _func_sum(x):
            return self.forward(x.reshape(shape), t).sum(dim=0).flatten()

        jacobian = torch.autograd.functional.jacobian(_func_sum, x.reshape(x.shape[0],-1), create_graph=create_graph).transpose(0,1)
        return torch.vmap(torch.trace)(jacobian).flatten()
    
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
        # steal MW's trick --> this assigns self.eges the first time.
        # Doesn't work for batches bcs batch size might very.
        if self.edges is None:
            self.edges = self._create_edges(n_particles)
        batches,edgesi,edgesj = self._cast_edges2batch(self.edges, n_batch, n_particles)
        edges = [edgesi.to(x.device), edgesj.to(x.device)]
        batches = batches.to(x.device)
        flat_coord = rearrange(x, 'b p d -> (b p) d')
        # compute pairwaise distances and direction information
        radial, coord_diff = self.coord2radial(edges, flat_coord) # one big matrix of size ((b p p) 1)
        # requires grad on radial
        radial.requires_grad_(True)

        
        # Compute S, dsdx, dWdx using compute_weights with gradients
        S = self.compute_weights(x, t, return_grad=True)
        divergence = torch.zeros(
            (n_batch, n_particles, x.shape[-1]),
            device=x.device,
            dtype=flat_features.dtype,
            requires_grad=False
        )
        # linear term
        divergence += reduce(
            S,  # (b, p, p, 1)
            'b p1 p2 d -> b p1 d',
            'sum'
        )
        # softmax term

        return

        # compute gradient of scale.sum() w.r.t. radial
        (dscale,) = torch.autograd.grad(
            outputs=scale.sum(),
            inputs=radial,
            create_graph=True
        )

        # pass features through the encoder
        flat_features = self.encoder(flat_features)  # ((b p p) 1)

        # compute gradient of flat_features.sum() w.r.t. radial
        (dfeat,) = torch.autograd.grad(
            outputs=flat_features.sum(),
            inputs=radial,
            create_graph=True
)


        # now we have to reshape to do a soft max operation.


        diffs = torch.zeros(
            (n_batch, n_particles, n_particles, x.shape[-1]),
            device=x.device,
            dtype=flat_features.dtype
        )
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

        #softmax
        sm = F.softmax(sm, dim=-2)

        # divergence vector to be filled


        #linear term

        divergence += reduce(
            sm * scales,  # (b, p, p, d),
            'b p1 p2 d -> b p1 d',
            'sum'
        )

        #scales term
        divergence += reduce(
            2 * diffs ** 2 * dscales * sm,
            'b p1 p2 d -> b p1 d',
            'sum'
        )

        #sm term

        # non cross terms
        sm_ = repeat(
            (- sm * dfeatures * 2 * diffs).sum(-2) ,
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
            (sm * scales * dfeatures * 2 *  diffs ** 2),
            'b p1 p2 d -> b p1 d',
            'sum'
            )

        divergence = reduce(
            divergence,
            'b p d -> b',
            'sum'
        )
        return divergence
