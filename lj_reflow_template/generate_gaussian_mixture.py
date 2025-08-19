import argparse
import math
import h5py
import torch

@torch.no_grad()
def logprob_wrapped_gmm(samples: torch.Tensor,
                        boxlength: float = 1.0,
                        means: torch.Tensor = None,
                        sigma: float = 0.1,
                        nwrap: int = 1) -> torch.Tensor:
    """
    Estimate log p(X) for a batch of samples under a wrapped 2D Gaussian mixture.

    Model (matches your sampler):
      For each point independently:
        1) choose a component k ~ Uniform({1..K})
        2) draw y ~ N(mu_k, sigma^2 I_2) in R^2
        3) wrap: x = ((y + L/2) mod L) - L/2

    p_wrap(x) = (1/K) * sum_k sum_{w in Z^2} N(x | mu_k + w*L, sigma^2 I)
    We approximate the infinite lattice sum with shifts in {-nwrap..nwrap}^2.

    Args:
        samples: (B, P, 2) tensor, coordinates in (-L/2, L/2]
        boxlength: L
        means: (K, 2) tensor of component centers (in same coords, i.e. (-L/2, L/2])
        sigma: std of the (unwrapped) Gaussians
        nwrap: how many image shells to include per axis (1 is usually enough when sigma << L)

    Returns:
        logp: (B,) tensor, total log-prob per configuration (sum over points)
    """
    if means is None:
        means = default_means(4, boxlength)
    B, P, D = samples.shape
    assert D == 2, "This function is for 2D."
    device = samples.device
    means = means.to(device)
    K = means.shape[0]

    # Ensure inputs are in the canonical cell (-L/2, L/2]
    L = float(boxlength)
    samples = (samples + 0.5 * L) % L - 0.5 * L
    means   = (means   + 0.5 * L) % L - 0.5 * L

    # Build wrap shift grid W = (2*nwrap+1)^2, shape (W, 2)
    shifts_1d = torch.arange(-nwrap, nwrap + 1, device=device, dtype=means.dtype)
    g_i, g_j = torch.meshgrid(shifts_1d, shifts_1d, indexing="ij")
    shifts = torch.stack([g_i, g_j], dim=-1).reshape(-1, 2) * L  # (W, 2)
    W = shifts.shape[0]

    # Broadcast shapes:
    # samples: (B, P, 1, 1, 2)
    # means:                 (K, 2) -> (1, 1, K, 1, 2)
    # shifts:                        (W, 2) -> (1, 1, 1, W, 2)
    x = samples.unsqueeze(2).unsqueeze(3)                 # (B, P, 1, 1, 2)
    mu = means.view(1, 1, K, 1, 2)                        # (1, 1, K, 1, 2)
    sh = shifts.view(1, 1, 1, W, 2)                       # (1, 1, 1, W, 2)

    # Compute log N(x | mu + shift, sigma^2 I)
    diff = x - (mu + sh)                                  # (B, P, K, W, 2)
    sq = (diff ** 2).sum(dim=-1)                          # (B, P, K, W)
    log_norm = - math.log(2 * math.pi * (sigma ** 2))     # 2D Gaussian
    log_pdf = log_norm - 0.5 * sq / (sigma ** 2)          # (B, P, K, W)

    # Sum over wrap images and components via log-sum-exp; include equal mixture weight 1/K
    log_pdf_k = torch.logsumexp(log_pdf, dim=3)           # (B, P, K)
    log_pdf_mix = torch.logsumexp(log_pdf_k, dim=2) - math.log(K)  # (B, P)

    # Total log-prob for each configuration: sum over all points
    logp = log_pdf_mix.sum(dim=1)                         # (B,)
    return logp

def default_means(n_peaks: int, boxlength: float) -> torch.Tensor:
    if n_peaks == 4:
        # corners of the box
        coords = torch.tensor([
            [-0.25, -0.25],
            [0.25, -0.25],
            [-0.25, 0.25],
            [0.25, 0.25],
        ]) * boxlength
    else:
        angles = torch.linspace(0, 2 * math.pi, n_peaks, endpoint=False)
        center = 0
        radius = boxlength / 4
        coords = torch.stack([
            center + radius * torch.cos(angles),
            center + radius * torch.sin(angles),
        ], dim=1)
    return coords

def sample_mixture(n_samples: int, npoints: int, boxlength: float, n_peaks: int, sigma: float) -> torch.Tensor:
    means = default_means(n_peaks, boxlength)
    print(f"Using means: {means.numpy()}")
    peak_idx = torch.randint(0, n_peaks, (n_samples, npoints))
    selected_means = means[peak_idx]
    samples = torch.randn(n_samples, npoints, 2) * sigma + selected_means
    samples = samples % boxlength - boxlength / 2 
    return samples

def main():
    parser = argparse.ArgumentParser(description="Generate 2D Gaussian mixture samples")
    parser.add_argument("out", type=str, help="Output HDF5 file path")
    parser.add_argument("--n_samples", type=int, default=100000, help="Number of samples to generate")
    parser.add_argument("--npoints", type=int, default=16, help="Number of points per sample")
    parser.add_argument("--boxlength", type=float, default=1.0, help="Box length for PBC")
    parser.add_argument("--n_peaks", type=int, default=4, help="Number of Gaussian peaks")
    parser.add_argument("--sigma", type=float, default=0.1, help="Standard deviation of Gaussians")
    args = parser.parse_args()

    data = sample_mixture(args.n_samples, args.npoints, args.boxlength, args.n_peaks, args.sigma)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("trajectory", data=data.numpy())

if __name__ == "__main__":
    main()
