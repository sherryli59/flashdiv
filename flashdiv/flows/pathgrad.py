import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


# ----------------------------
# Autograd helpers
# ----------------------------

def grad_divergence(z, t, f, probe):
    """gdiv = ∇_x tr J_v via Hutchinson using the SAME probe."""
    with torch.enable_grad():
        z_req = z.detach().requires_grad_(True)
        fz = f(z_req, t)
        (JTe,) = torch.autograd.grad(
            outputs=fz, inputs=z_req, grad_outputs=probe,
            create_graph=True, retain_graph=False
        )
        div_est = (JTe * probe).sum(dim=-1)
        (gdiv,) = torch.autograd.grad(
            outputs=div_est.sum(), inputs=z_req,
            create_graph=False, retain_graph=False
        )
        return gdiv

def JT_v(v: torch.Tensor, z: torch.Tensor, t: torch.Tensor, f) -> torch.Tensor:
    """Return J_f(z,t)^T @ v via autograd (no graph needed beyond this)."""
    with torch.enable_grad():
        z_req = z.detach().requires_grad_(True)
        fz = f(z_req, t)  # (B,D)
        (JT_v_val,) = torch.autograd.grad(
            outputs=fz, inputs=z_req, grad_outputs=v,
            create_graph=False, retain_graph=False
        )
    return JT_v_val


# ----------------------------
# Velocity wrapper
# ----------------------------

class FlowFunc(nn.Module):
    def __init__(self, vnet):
        super().__init__()
        self.vnet = vnet
    def forward(self, t, z: torch.Tensor) -> torch.Tensor:
        # t: scalar or (B,), z: (B,D) or (B,*,D) — assume last dim are coords
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=z.device, dtype=z.dtype)
        else:
            t = t.to(device=z.device, dtype=z.dtype)
        if t.dim() == 0:
            t = t.expand(z.size(0))
        return self.vnet(z, t)


# ----------------------------
# Divergence estimator (Hutchinson)
# ----------------------------

def divergence_hutchinson(z, t, f, probe=None, *, create_graph: bool):
    """
    Estimate tr J_v(z,t) = E[ ξ^T J ξ ]. If create_graph=True, the result
    keeps a graph w.r.t. z so logq1 depends on x0 (needed for Variant B).
    """
    with torch.enable_grad():
        if probe is None:
            probe = (torch.randint(0, 2, z.shape, device=z.device, dtype=z.dtype) * 2 - 1)
        z_req = z.detach().requires_grad_(True)
        fz = f(z_req, t)
        (JTe,) = torch.autograd.grad(
            outputs=fz, inputs=z_req, grad_outputs=probe,
            create_graph=create_graph, retain_graph=False
        )
        div_est = (JTe * probe).sum(dim=-1)
        return div_est, probe


# ----------------------------
# CNF for (x, logq) forward integration
# ----------------------------

class CNFLogProbFunc(nn.Module):
    """
    ODE on the augmented state (x, logq), where
      dx/dt   = vθ(x,t)
      d/dt logq = - tr J_v(x,t)
    Uses Hutchinson’s estimator for the divergence.
    """
    def __init__(self, vnet, div_samples: int = 1):
        super().__init__()
        self.vnet = vnet
        self.div_samples = int(div_samples)

    def forward(self, t, state):
        x, logq = state
        # standardize time to (B,)
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        else:
            t = t.to(device=x.device, dtype=x.dtype)
        t_vec = t.expand(x.size(0)) if t.dim() == 0 else t

        v = self.vnet(x, t_vec)
        # Build divergence with graph so logq depends on x
        div_sum = 0.0
        for _ in range(self.div_samples):
            div_est, _ = divergence_hutchinson(x, t_vec, lambda Z, T: self.vnet(Z, T),
                                               probe=None, create_graph=True)
            div_sum = div_sum + div_est
        div = div_sum / float(self.div_samples)   # (B,)
        dlogq = -div
        return v, dlogq


def forward_x_and_logq(x0: torch.Tensor, vnet, t_grid: torch.Tensor, *, div_samples=1,
                       method='dopri5', rtol=1e-5, atol=1e-6):
    """
    Single forward pass that returns x1, logq1 WITH a graph so that
    logq1 depends on x0 (needed to recover ∇_{x1} log q via JT solve).
    """
    func = CNFLogProbFunc(vnet, div_samples=div_samples)
    logq0 = torch.zeros(x0.size(0), device=x0.device, dtype=x0.dtype)
    x_path, logq_path = odeint(
        func, (x0, logq0), t_grid,
        method=method, rtol=rtol, atol=atol, 
    )
    return x_path[-1], logq_path[-1]   # x1, logq1


# ----------------------------
# Reverse-KL path gradient (Variant B, scalarizer)
# ----------------------------

def reverse_kl_pathgrad_loss(vnet, target_dist, 
                             x0: torch.Tensor,
                             t_grid: torch.Tensor,
                             div_samples: int = 3,
                             method: str = 'dopri5',
                             rtol: float = 1e-5,
                             atol: float = 1e-6):
    """
    Implements: H = ∇E(x1) + ∇_x log qθ(x1), φ = <x1, stopgrad(H)>, loss = φ (+ detached logs)
    Returns: loss (scalar), logging dict
    """
    device = next(vnet.parameters()).device


    # 2) forward once with a graph to get x1, logq1
    x1, logq1 = forward_x_and_logq(x0, vnet, t_grid, div_samples=div_samples,
                                   method=method, rtol=rtol, atol=atol)

    # 3) gradE = ∇E(x1)
    gradE = -target_dist.force(x1)  
    # 4) ∇_{x1} log qθ(x1) via two-pass JT solve:
    #    compute g_x0 = ∇_{x0} logq1
    (g_x0,) = torch.autograd.grad(logq1.sum(), x0, retain_graph=True, create_graph=True)

    #    build J^T columns for the coordinate basis (last dim = D)
    D = x1.shape[-1]
    basis = torch.eye(D, device=device, dtype=x1.dtype)  # (D,D)
    cols = []
    for d in range(D):
        e = basis[d].view(*(([1] * (x1.dim() - 1)) + [D]))  # shape like x1 per sample
        e = e.expand_as(x1)
        (JTcd,) = torch.autograd.grad(x1, x0, grad_outputs=e, retain_graph=True, create_graph=True)
        cols.append(JTcd)  # each (B, *coords)
    JT = torch.stack(cols, dim=-1)  # (..., D) stacked at the end → J^T

    B = x1.shape[0]
    JT_mat = JT.reshape(B, -1, D)            # (B, n, D) == J^T
    g_x0_vec = g_x0.reshape(B, -1, 1)        # (B, n, 1)
    g_x1 = torch.linalg.lstsq(JT_mat, g_x0_vec).solution.squeeze(-1)  # (B, D)
    g_x1 = g_x1.reshape_as(x1)

    # 5) cotangent H and scalarizer φ
    H = (gradE + g_x1).detach()
    phi = (x1 * H).reshape(B, -1).sum(dim=1).mean()

    # 6) (optional) logging objective value L = E(x1) + logq1
    with torch.no_grad():
        L = (-target_dist.log_prob(x1) + logq1).mean()

    # only φ carries gradients; add detached L for a stable logging scalar if desired
    loss = L.detach() + (phi - phi.detach())
    logs = {"train/L": L.detach(), "train/phi": phi.detach()}
    return loss, logs
