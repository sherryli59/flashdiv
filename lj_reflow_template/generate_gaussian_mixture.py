import argparse
import h5py
import torch
from pytorch_lightning import seed_everything
import math

def wrap_to_cell(x: torch.Tensor, L: float) -> torch.Tensor:
    """Map coordinates to the canonical cell (-L/2, L/2]."""
    return (x + 0.5 * L) % L - 0.5 * L

def generate_means(dim: int, n_peaks: int, boxlength: float) -> torch.Tensor:
    """
    Choose `n_peaks` means from all 2**dim corners at ±L/4 in each dim.
    Returns (n_peaks, dim).
    """
    base = torch.tensor([-0.25, 0.25], dtype=torch.float32) * float(boxlength)
    all_means = torch.cartesian_prod(*([base] * dim))  # (2**dim, dim)
    if n_peaks > all_means.shape[0]:
        raise ValueError(
            f"n_peaks={n_peaks} exceeds number of possible peaks {all_means.shape[0]}"
        )
    idx = torch.randperm(all_means.shape[0])[:n_peaks]
    means = all_means[idx]
    # ensure means lie in canonical cell (they already do, but be explicit)
    return wrap_to_cell(means, boxlength)

def sample_mixture(
    n_samples: int,
    dim: int = 16,
    n_peaks: int = 20,
    sigma: float = 0.1,
    boxlength: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate samples from an equal-weight isotropic Gaussian mixture on a torus.

    Means are chosen among all ±L/4 corners in each dimension, and samples are
    wrapped into (-L/2, L/2]. Returns:
      samples: (n_samples, dim, 1)
      means:   (n_peaks, dim)
    """
    L = float(boxlength)
    means = generate_means(dim, n_peaks, L)
    print(f"Using means (first up to 10 shown):\n{means[:10].numpy()}")

    peak_idx = torch.randint(0, n_peaks, (n_samples,))
    selected_means = means[peak_idx]                       # (n_samples, dim)
    samples = torch.randn(n_samples, dim) * sigma + selected_means
    samples = wrap_to_cell(samples, L)                     # torus wrapping
    return samples.unsqueeze(-1), means

@torch.no_grad()
def logprob_mixture_torus(
    samples: torch.Tensor,
    *,
    boxlength: float,
    means: torch.Tensor | None = None,
    n_peaks: int = 10,
    dim: int = 16,
    sigma: float = 0.1,
    nwrap: int = 2,
) -> torch.Tensor:
    """
    Log p(x) under an equal-weight Gaussian mixture **on a torus**.

    Exact periodization (per-dimension) via truncated lattice sum:
        p(x | mu) = Prod_d [ (1/(sqrt(2π)σ)) * Σ_{m∈Z} exp(-(x_d - mu_d - mL)^2/(2σ^2)) ]
    We truncate m to {-nwrap,...,nwrap}. Then mix equally over K components.

    Args:
        samples: (B, D, 1) or (B, D), coordinates in (-L/2, L/2].
        boxlength: L (float)
        means: (K, D) component centers; if None, they are generated randomly
               with `generate_means(dim, n_peaks, boxlength)`.
        n_peaks, dim, sigma: mixture/model params (only used if means is None).
        nwrap: truncation radius for the lattice sum per dimension.

    Returns:
        (B,) log-probabilities.
    """
    L = float(boxlength)
    if samples.dim() == 3 and samples.size(-1) == 1:
        samples = samples.squeeze(-1)  # (B, D)
    samples = wrap_to_cell(samples, L)

    if means is None:
        means = generate_means(dim, n_peaks, L)
    means = wrap_to_cell(means, L)     # (K, D)

    device = samples.device
    dtype = samples.dtype
    means = means.to(device=device, dtype=dtype)

    B, D = samples.shape
    K, Dm = means.shape
    assert D == Dm, f"Dim mismatch: samples D={D}, means D={Dm}"

    s2 = float(sigma) ** 2
    const = -0.5 * D * math.log(2.0 * math.pi * s2)

    # diffs: (B, K, D)
    diffs = samples.unsqueeze(1) - means.unsqueeze(0)  # x - mu

    # Lattice shifts per-dimension (M = 2*nwrap+1)
    mvals = torch.arange(-nwrap, nwrap + 1, device=device, dtype=dtype) * L  # (M,)

    # For each dimension, compute log Σ_m exp(- (diff - mL)^2 / (2σ^2))
    # Shape bookkeeping:
    # diffs[..., None] -> (B, K, D, 1), broadcast minus (M,) -> (B, K, D, M)
    shifted = diffs.unsqueeze(-1) - mvals.view(1, 1, 1, -1)  # (B, K, D, M)
    exp_terms = -0.5 * (shifted ** 2) / s2                   # (B, K, D, M)
    logsum_per_dim = torch.logsumexp(exp_terms, dim=-1)      # (B, K, D)

    # Product over dimensions -> sum of logs over dims, add Gaussian norm const
    log_pdf_given_k = const + logsum_per_dim.sum(dim=-1)     # (B, K)

    # Equal-weight mixture over K components
    logp = torch.logsumexp(log_pdf_given_k, dim=1) - math.log(K)  # (B,)
    return logp

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate multi-dimensional Gaussian mixture samples on a torus"
    )
    parser.add_argument("out", type=str, help="Output HDF5 file path")
    parser.add_argument("--n_samples", type=int, default=100000, help="Number of samples to generate")
    parser.add_argument("--dim", type=int, default=16, help="Dimensionality (number of 1D points)")
    parser.add_argument("--n_peaks", type=int, default=10, help="Number of Gaussian peaks (components)")
    parser.add_argument("--sigma", type=float, default=0.1, help="Std of Gaussians")
    parser.add_argument("--boxlength", type=float, default=1.0, help="Box length L for periodic BCs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--compute_logprob", action="store_true", help="Compute and store logprob for generated samples")
    parser.add_argument("--nwrap", type=int, default=1, help="Lattice truncation radius for wrapped pdf")
    args = parser.parse_args()

    seed_everything(args.seed)

    data, means = sample_mixture(args.n_samples, args.dim, args.n_peaks, args.sigma, args.boxlength)
    print("samples shape:", data.shape)

    with h5py.File(args.out, "w") as f:
        f.create_dataset("trajectory", data=data.numpy())        # (N, D, 1)
        f.create_dataset("means", data=means.numpy())            # (K, D)
        f.attrs["boxlength"] = float(args.boxlength)
        f.attrs["sigma"] = float(args.sigma)
        f.attrs["n_peaks"] = int(args.n_peaks)
        f.attrs["dim"] = int(args.dim)
        if args.compute_logprob:
            # Compute log-prob for the generated batch (may be large -> chunk if needed)
            batch = data.squeeze(-1)                             # (N, D)
            logp = logprob_mixture_torus(
                batch, boxlength=args.boxlength, means=means, sigma=args.sigma, nwrap=args.nwrap
            )
            f.create_dataset("logprob", data=logp.cpu().numpy()) # (N,)

if __name__ == "__main__":
    main()
