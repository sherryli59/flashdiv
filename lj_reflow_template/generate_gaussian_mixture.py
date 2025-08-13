import argparse
import math
import h5py
import torch

def default_means(n_peaks: int, boxlength: float) -> torch.Tensor:
    if n_peaks == 4:
        # corners of the box
        coords = torch.tensor([
            [0.25, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.75, 0.75],
        ]) * boxlength
    else:
        angles = torch.linspace(0, 2 * math.pi, n_peaks, endpoint=False)
        center = boxlength / 2
        radius = boxlength / 4
        coords = torch.stack([
            center + radius * torch.cos(angles),
            center + radius * torch.sin(angles),
        ], dim=1)
    return coords

def sample_mixture(n_samples: int, npoints: int, boxlength: float, n_peaks: int, sigma: float) -> torch.Tensor:
    means = default_means(n_peaks, boxlength)
    peak_idx = torch.randint(0, n_peaks, (n_samples, npoints))
    selected_means = means[peak_idx]
    samples = torch.randn(n_samples, npoints, 2) * sigma + selected_means
    samples = samples % boxlength
    return samples

def main():
    parser = argparse.ArgumentParser(description="Generate 2D Gaussian mixture samples")
    parser.add_argument("out", type=str, help="Output HDF5 file path")
    parser.add_argument("--n_samples", type=int, default=100000, help="Number of samples to generate")
    parser.add_argument("--npoints", type=int, default=16, help="Number of points per sample")
    parser.add_argument("--boxlength", type=float, default=1.0, help="Box length for PBC")
    parser.add_argument("--n_peaks", type=int, default=4, help="Number of Gaussian peaks")
    parser.add_argument("--sigma", type=float, default=0.2, help="Standard deviation of Gaussians")
    args = parser.parse_args()

    data = sample_mixture(args.n_samples, args.npoints, args.boxlength, args.n_peaks, args.sigma)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("trajectory", data=data.numpy())

if __name__ == "__main__":
    main()
