import argparse
import h5py
import torch
from pytorch_lightning import seed_everything


def sample_mixture(
    n_samples: int,
    dim: int = 16,
    n_peaks: int = 20,
    sigma: float = 0.1,
) -> torch.Tensor:
    """Generate samples from a multi-dimensional Gaussian mixture.

    There are ``2**dim`` possible component means given by all combinations of
    ``±0.25`` in each dimension.  We randomly choose ``n_peaks`` of these
    combinations (without replacement) to form the mixture components.  Each
    selected component has isotropic covariance ``sigma**2 I`` and all
    components are weighted equally.

    Args:
        n_samples: Number of samples to draw.
        dim: Dimensionality of each sample.
        n_peaks: Number of mixture components to sample.
        sigma: Standard deviation of each Gaussian component.

    Returns:
        Tensor of shape ``(n_samples, dim)`` containing the generated samples.
    """

    # Enumerate all possible means located at ±0.25 in each dimension
    all_means = torch.cartesian_prod(*([torch.tensor([-0.25, 0.25])] * dim))
    if n_peaks > all_means.shape[0]:
        raise ValueError(
            f"n_peaks={n_peaks} exceeds number of possible peaks {all_means.shape[0]}"
        )

    # Choose a subset of peaks uniformly at random
    idx = torch.randperm(all_means.shape[0])[:n_peaks]
    means = all_means[idx]
    print(f"Using means: {means.numpy()}")

    # Sample from the selected mixture components
    peak_idx = torch.randint(0, n_peaks, (n_samples,))
    selected_means = means[peak_idx]
    samples = torch.randn(n_samples, dim) * sigma + selected_means
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate multi-dimensional Gaussian mixture samples"
    )
    parser.add_argument("out", type=str, help="Output HDF5 file path")
    parser.add_argument(
        "--n_samples", type=int, default=100000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--dim", type=int, default=16, help="Dimensionality of each sample"
    )
    parser.add_argument(
        "--n_peaks", type=int, default=20, help="Number of Gaussian peaks"
    )
    parser.add_argument(
        "--sigma", type=float, default=0.1, help="Standard deviation of Gaussians"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    data = sample_mixture(args.n_samples, args.dim, args.n_peaks, args.sigma)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("trajectory", data=data.numpy())


if __name__ == "__main__":
    main()

