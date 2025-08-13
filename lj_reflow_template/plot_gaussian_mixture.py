import argparse
import h5py
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot distribution of points from Gaussian mixture HDF5 file"
    )
    parser.add_argument("infile", type=str, help="Path to input HDF5 file")
    parser.add_argument(
        "--out",
        type=str,
        default="gaussian_mixture.png",
        help="Output image filename",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot instead of saving",
    )
    args = parser.parse_args()

    with h5py.File(args.infile, "r") as f:
        traj = f["trajectory"][:]

    points = traj.reshape(-1, 2)

    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], s=1, alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gaussian mixture sample distribution")

    if args.show:
        plt.show()
    else:
        plt.savefig(args.out, dpi=300)
        print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
