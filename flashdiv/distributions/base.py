import math
from typing import Optional

import torch


def _even_spacing(num_particles: int, boxlength: float, dim: int, *, device=None, dtype=None) -> torch.Tensor:
    """Construct evenly spaced particle positions inside the box."""
    device = device or torch.device("cpu")
    dtype = dtype or torch.get_default_dtype()
    positions = torch.zeros((num_particles, dim), device=device, dtype=dtype)
    num_per_dim = int(math.ceil(num_particles ** (1.0 / dim)))
    spacing = boxlength / num_per_dim

    idx = 0
    if dim == 3:
        for i in range(num_per_dim):
            for j in range(num_per_dim):
                for k in range(num_per_dim):
                    if idx >= num_particles:
                        return positions
                    positions[idx] = torch.tensor([i, j, k], device=device, dtype=dtype) * spacing
                    idx += 1
    elif dim == 2:
        for i in range(num_per_dim):
            for j in range(num_per_dim):
                if idx >= num_particles:
                    return positions
                positions[idx] = torch.tensor([i, j], device=device, dtype=dtype) * spacing
                idx += 1
    else:
        raise ValueError("_even_spacing only supports dim=2 or dim=3")
    return positions


def _wrapped_normal_logpdf(x, mu, sigma, boxlength, K=3):
    sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
    boxlength = torch.as_tensor(boxlength, dtype=x.dtype, device=x.device)

    x, mu, sigma, boxlength = torch.broadcast_tensors(x, mu, sigma, boxlength)
    ks = torch.arange(-K, K + 1, dtype=x.dtype, device=x.device)

    shifted = x.unsqueeze(-1) + ks * boxlength.unsqueeze(-1)
    log_gauss = (
        -0.5 * ((shifted - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)) ** 2
        - torch.log(sigma.unsqueeze(-1))
        - 0.5 * torch.log(torch.tensor(2 * torch.pi, dtype=x.dtype, device=x.device))
    )
    return torch.logsumexp(log_gauss, dim=-1) - torch.log(boxlength)


class Uniform:
    def __init__(self, boxlength: float, nparticles: int, dim: int):
        self.boxlength = float(boxlength)
        self.nparticles = int(nparticles)
        self.dim = int(dim)
        self._device = torch.device("cpu")
        self._dtype = torch.get_default_dtype()

    def to(self, device, dtype: Optional[torch.dtype] = None):
        self._device = torch.device(device)
        self._dtype = dtype or self._dtype
        return self

    def sample(self, n: int) -> torch.Tensor:
        L = torch.as_tensor(self.boxlength, device=self._device, dtype=self._dtype)
        samples = torch.rand((n, self.nparticles, self.dim), device=self._device, dtype=self._dtype)
        samples = samples * L - L / 2
        return samples

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        L = torch.as_tensor(self.boxlength, device=x.device, dtype=x.dtype)
        exponent = self.nparticles * self.dim
        log_density = -exponent * torch.log(L)
        return torch.full((x.size(0),), log_density, device=x.device, dtype=x.dtype)


    def score(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class WrappedGaussian:
    def __init__(self, boxlength: float, nparticles: int, dim: int, std: float = 0.5):
        self.boxlength = float(boxlength)
        self.nparticles = int(nparticles)
        self.dim = int(dim)
        self.std = float(std)
        self._device = torch.device("cpu")
        self._dtype = torch.get_default_dtype()
        self._mean = None

    def to(self, device, dtype: Optional[torch.dtype] = None):
        self._device = torch.device(device)
        self._dtype = dtype or self._dtype
        self._mean = _even_spacing(
            self.nparticles,
            self.boxlength,
            self.dim,
            device=self._device,
            dtype=self._dtype,
        )
        return self

    @property
    def mean(self) -> torch.Tensor:
        if self._mean is None:
            self.to(self._device, self._dtype)
        return self._mean

    def sample(self, n: int) -> torch.Tensor:
        mean = self.mean
        cov = (self.std ** 2) * torch.eye(self.nparticles * self.dim, device=mean.device, dtype=self._dtype)
        dist = torch.distributions.MultivariateNormal(mean.flatten(), covariance_matrix=cov)
        flat = dist.sample((n,))
        return flat.view(n, self.nparticles, self.dim).remainder(self.boxlength)

    def sample_with_log_prob(self, n: int):
        samples = self.sample(n)
        logp = self.log_prob(samples)
        return samples, logp

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        mu = self.mean.expand(x.size(0), -1, -1).to(x.device, x.dtype)
        logp = _wrapped_normal_logpdf(x, mu, self.std, self.boxlength)
        return logp.sum(dim=(-1, -2))

    def score(self, x: torch.Tensor) -> torch.Tensor:
        x_req = x.detach().requires_grad_(True)
        logp = self.log_prob(x_req)
        grad = torch.autograd.grad(logp.sum(), x_req)[0]
        return grad.detach()


class SimpleGaussian:
    """Unwrapped diagonal Gaussian base distribution on R^{nparticles x dim}."""

    def __init__(self, nparticles: int, dim: int, std: float = 1.0, mean: float = 0.0):
        self.nparticles = int(nparticles)
        self.dim = int(dim)
        self.std = float(std)
        self.mean = float(mean)
        self._device = torch.device("cpu")
        self._dtype = torch.get_default_dtype()

    def to(self, device, dtype: Optional[torch.dtype] = None):
        self._device = torch.device(device)
        self._dtype = dtype or self._dtype
        return self

    def _broadcast_params(self):
        shape = (self.nparticles, self.dim)
        mean = torch.full(shape, self.mean, device=self._device, dtype=self._dtype)
        std = torch.full(shape, self.std, device=self._device, dtype=self._dtype)
        return mean, std

    def sample(self, n: int) -> torch.Tensor:
        mean, std = self._broadcast_params()
        noise = torch.randn((n, self.nparticles, self.dim), device=self._device, dtype=self._dtype)
        return noise * std + mean

    def sample_with_log_prob(self, n: int):
        samples = self.sample(n)
        logp = self.log_prob(samples)
        return samples, logp

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self._broadcast_params()
        mean = mean.to(x.device, x.dtype)
        std = std.to(x.device, x.dtype)
        var = std ** 2
        norm_const = -0.5 * math.log(2 * math.pi)
        scaled = ((x - mean) ** 2) / var
        log_det = torch.log(std)
        logp = norm_const - log_det - 0.5 * scaled
        return logp.sum(dim=(-1, -2))

    def score(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self._broadcast_params()
        mean = mean.to(x.device, x.dtype)
        std = std.to(x.device, x.dtype)
        return -(x - mean) / (std ** 2)
