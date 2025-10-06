import torch

class BaseDistribution(torch.nn.Module):
    def __init__(self, nparticles=4, dim=2, batch_size=10, device='cuda'):
        super(BaseDistribution, self).__init__()
        self.batch_size = batch_size
        self.nparticles = nparticles
        self.dim = dim
        self.param = torch.nn.Parameter(torch.randn(batch_size,nparticles,dim).to(device))

    @property
    def state(self):
        return self.param

    def potential(self, x):
        pass

    def forward(self, x=None):
        return self.log_prob(x)

    def log_prob(self, x=None):
        if x is None:
            x = self.param
        return -self.potential(x)

    def grad_log_prob(self, x=None):
        if x is None:
            x = self.param
        x.requires_grad_(True)
        with torch.enable_grad():
            return  -torch.autograd.grad(-self.potential(x), x,
                                         torch.ones(x.shape[0]).to(x.device), create_graph=True)[0]

    def neg_force_clipped(self, x=None, max_val=80):
        if x is None:
            x = self.param
        return torch.clip(self.grad_log_prob(x),-max_val,max_val)

    def reset_parameters(self):
        self.param.data = torch.randn(self.batch_size,self.nparticles,self.dim).to(self.param.device)

class DW(BaseDistribution):
    """
    Double-Well pairwise distance potential (DW-4), generalizable to any (nparticles, dim).
    Default parameters match the paper snippet: a=0, b=-4, c=0.9, tau=1.
    E(x) = sum_{i<j} [ a*(d_ij - d0) + b*(d_ij - d0)^2 + c*(d_ij - d0)^4 ]
    where d_ij = ||x_i - x_j||_2
    """
    def __init__(self, nparticles=4, dim=2, batch_size=10, device="cuda",
                 a=0.0, b=-4.0, c=0.9, d0=4.0, kT=1.0,
                 periodic=False, boxlength=None, spring_constant=0.0):
        super().__init__(nparticles=nparticles, dim=dim, batch_size=batch_size, device=device)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d0 = float(d0)
        self.kT = float(kT)  # temperature scale
        self.periodic = periodic
        self.boxlength = boxlength
        self.spring_constant = float(spring_constant)  # optional COM tether (0 disables)

    # ---------- core potential ----------
    def potential(self, x):
        """
        Returns total potential energy per batch element (shape: [batch]).
        NOTE: This returns the *energy* (no 1/tau factor). Use log_likelihood() for heat version.
        """
        # x: [B, N, D]
        pair_vec = self._pair_vec(x)                     # [B, N, N, D]
        dij = torch.linalg.norm(pair_vec, dim=-1)        # [B, N, N]

        # Use only upper triangle (i<j) to avoid double-counting/self-pairs:
        N = dij.shape[-1]
        iu, ju = torch.triu_indices(N, N, offset=1, device=dij.device)
        d = dij[..., iu, ju]                             # [B, N_pairs]

        r = d - self.d0
        e_pairs = self.a * r + self.b * r.pow(2) + self.c * r.pow(4)  # [B, N_pairs]
        E = e_pairs.sum(dim=-1)                          # [B]

        # Optional harmonic tether to keep system centered if desired:
        if self.spring_constant > 0.0 and not self.periodic:
            E = E + self._harmonic_potential(x)
        return E

    def log_likelihood(self, x):
        """
        Heat-version log probability: -E / tau (like your LJ.log_likelihood).
        """
        return - self.potential(x) / self.tau

    # ---------- helpers ----------
    def _pair_vec(self, x):
        # pairwise displacement with optional PBC
        pv = x.unsqueeze(-2) - x.unsqueeze(-3)  # [B, N, N, D]
        if self.periodic and (self.boxlength is not None):
            # minimum-image convention; supports scalar or per-dimension boxlength
            L = self.boxlength
            pv = pv - torch.round(pv / L) * L
        return pv

    def _harmonic_potential(self, x):
        # 0.5 * k * sum_i ||x_i - COM||^2
        com = x.mean(dim=-2, keepdim=True)            # [B, 1, D]
        rel = x - com                                 # [B, N, D]
        return 0.5 * self.spring_constant * (rel**2).sum(dim=(-1, -2))  # [B]
    
    def force(self, x=None, eps: float = 1e-12):
        """
        Total force on each particle: shape [B, N, D].
        For each pair (i,j):  E_ij = a*(d-d0) + b*(d-d0)^2 + c*(d-d0)^4,  d=||x_i-x_j||.
        dE/dd = a + 2b*(d-d0) + 4c*(d-d0)^3
        F_i = -∑_j (dE/dd) * (x_i - x_j) / d
        """
        if x is None:
            x = self.param

        # Pair displacements (minimum image if periodic)
        pv = self._pair_vec(x)                               # [B, N, N, D]
        dij = torch.linalg.norm(pv, dim=-1)                  # [B, N, N]
        d_safe = dij.clamp_min(eps)

        # dE/dd for each pair
        r = dij - self.d0
        g = self.a + 2.0*self.b*r + 4.0*self.c*r.pow(3)      # [B, N, N]
        g_over_d = g / d_safe                                # [B, N, N]
        g_over_d = g_over_d.masked_fill(dij <= eps, 0.0)     # zero the diagonal

        # Pairwise force on i due to j:  -g(d)/d * (x_i - x_j)
        pair_force = -(g_over_d.unsqueeze(-1)) * pv          # [B, N, N, D]

        # Sum over neighbors j to get net force on each i
        total_force = pair_force.sum(dim=-2)                 # [B, N, D]

        # Optional COM tether only for non-periodic systems
        if (not self.periodic) and (self.spring_constant > 0.0):
            total_force = total_force + self._harmonic_force(x)

        return total_force

    def _harmonic_force(self, x):
        com = x.mean(dim=-2, keepdim=True)                   # [B, 1, D]
        rel = x - com
        return -self.spring_constant * rel                   # [B, N, D]