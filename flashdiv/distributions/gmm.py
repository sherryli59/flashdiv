import torch
# -----------------------------
# 2D Gaussian Mixture target
# -----------------------------
class GMM2D:
    def __init__(self, means, covs, weights):
        self.means = [torch.tensor(m, dtype=torch.get_default_dtype()) for m in means]
        self.covs = [torch.tensor(C, dtype=torch.get_default_dtype()) for C in covs]
        self.weights = torch.tensor(weights, dtype=torch.get_default_dtype())
        self.K = len(means)
        self._mvns = [torch.distributions.MultivariateNormal(self.means[k], self.covs[k]) for k in range(self.K)]

    def to(self, device, dtype=None):
        self.means   = [m.to(device=device, dtype=dtype or m.dtype) for m in self.means]
        self.covs    = [C.to(device=device, dtype=dtype or C.dtype) for C in self.covs]
        self.weights = self.weights.to(device=device, dtype=dtype or self.weights.dtype)
        self._mvns   = [torch.distributions.MultivariateNormal(self.means[k], self.covs[k])
                    for k in range(self.K)]
        return self
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        comps = torch.stack([self._mvns[k].log_prob(x) + self.weights[k].log() for k in range(self.K)], dim=0)
        return torch.logsumexp(comps, dim=0)
    
    def score(self, x: torch.Tensor) -> torch.Tensor:
        x_req = x.detach().requires_grad_(True)
        lp = self.log_prob(x_req).sum()
        (grad_x,) = torch.autograd.grad(lp, x_req)
        return grad_x.detach()

    def force(self, x: torch.Tensor) -> torch.Tensor:
        return self.score(x)
    
    def sample(self, n: int) -> torch.Tensor:
        w = self.weights
        cat = torch.distributions.Categorical(w / w.sum())
        comp_idx = cat.sample((n,))             # (n,)
        xs = []
        for k in range(self.K):
            nk = (comp_idx == k).sum().item()
            if nk == 0: continue
            xs.append(self._mvns[k].sample((nk,)))
        if len(xs) == 0:
            # very unlikely, but fallback
            xs = [self._mvns[0].sample((n,))]
        x = torch.cat(xs, dim=0)
        # shuffle to break component grouping
        perm = torch.randperm(x.size(0), device=x.device)
        return x[perm]

