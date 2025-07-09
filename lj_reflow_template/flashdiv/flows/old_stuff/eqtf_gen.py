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

    def compute_weights(self, x, t):
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
        r_ik = R.unsqueeze(3).expand(-1, -1, -1, P, -1)   # (B,P,P,P,1)
    
        # time broadcast
        t4 = t.view(B, 1, 1, 1, 1).expand(-1, P, P, P, 1)

        # concatenate  →  (B,P,P,P,4)
        triplet_feat = torch.cat((r_ij, r_jk, r_ik, t4), dim=-1)

        # three-body scale  s_{ijk}
        s = self.scaler(triplet_feat)                            # (B,P,P,P,1)
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

    