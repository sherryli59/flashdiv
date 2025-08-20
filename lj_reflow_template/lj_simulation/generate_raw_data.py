import torch
import argparse
import numpy as np
import h5py
from lj import LJ
from sampling import even_spacing
import matplotlib.pyplot as plt


def run_MH(target, x_init, n_steps, dt, adaptive=True, save_every=100, burn_in=1000, xs=None):
    '''
    MH sampler with burn-in, adaptive dt, and HDF5 storage.
    '''
    x = x_init.clone()
    acc = 0
    pot_list = []
    idx = 0

    for i in range(n_steps + burn_in):
        x_prop = x.clone()

        # Propose: move one particle per chain
        particle_idx = torch.randint(0, x.shape[1], (x.shape[0],))
        noise = dt * torch.randn_like(x[:, 0, :])
        x_prop[torch.arange(x.shape[0]), particle_idx] += noise

        # Metropolis ratio
        potential_x = target.potential(x)
        potential_prop = target.potential(x_prop)
        ratio = (potential_x - potential_prop) / target.kT
        accept_prob = torch.exp(ratio)
        u = torch.rand_like(accept_prob)
        acc_i = u < torch.minimum(accept_prob, torch.ones_like(accept_prob))

        # Accept/reject
        x[acc_i] = x_prop[acc_i]
        acc += acc_i.float().mean()

        # Burn-in plotting
        if i < burn_in:
            pot_list.append(potential_x.mean().item())

        # Save samples
        if i >= burn_in and (i - burn_in) % save_every == 0:
            xs[idx, ...] = x.detach().cpu().numpy()
            idx += 1

        # Adapt step size
        if i < burn_in and i % save_every == 0:
            acc_rate = acc / (i + 1)
            if adaptive:
                if acc_rate < 0.3:
                    dt *= 1 - 0.5 * (0.3 - acc_rate.item())
                elif acc_rate > 0.5:
                    dt *= 1 + 0.5 * (acc_rate.item() - 0.5)
            print(f"Step {i}, acc_rate {acc_rate:.3f}, dt {dt:.6f}")

        # # Burn-in plot
        # if i == burn_in:
        #     pot_list = torch.tensor(pot_list) / 500
        #     plt.plot(pot_list.detach().cpu().numpy())
        #     plt.savefig("pot_burn_in_mh.png")
        #     plt.close()

    return xs, acc / n_steps

def run_MALA(target, x_init, n_steps, dt, adaptive=True,
             save_every=100, burn_in=1000, xs=None, center_com=False):
    '''
    MALA sampler with burn-in, adaptive dt, and HDF5 storage.
    '''
    print("Running MALA")
    acc = 0
    idx = 0
    pot_list = []
    force_current = target.force(x_init)
    potential_current = target.potential(x_init)
    x = x_init

    for i in range(n_steps + burn_in):
        x_new = x + dt * force_current
        x_new += np.sqrt(2 * dt * target.kT) * torch.randn_like(x)

        potential_new = target.potential(x_new)

        if i < burn_in:
            pot_list.append(potential_current[0].clone())


        force_new = target.force(x_new)

        # Metropolis-Hastings acceptance
        ratio = potential_current - potential_new
        ratio -= ((x - x_new - dt * force_new) ** 2 / (4 * dt)).sum((1, 2))
        ratio += ((x_new - x - dt * force_current) ** 2 / (4 * dt)).sum((1, 2))
        ratio = ratio / target.kT
        ratio = torch.exp(ratio)

        u = torch.rand_like(ratio)
        acc_i = u < torch.min(ratio, torch.ones_like(ratio))

        # Accept/reject
        x[acc_i] = x_new[acc_i]
        force_current[acc_i] = force_new[acc_i]
        potential_current[acc_i] = potential_new[acc_i]
        print(potential_current)
        acc += acc_i.float().mean()

        # Save samples
        if i >= burn_in and (i - burn_in) % save_every == 0:
            if center_com:
                com = x.mean(dim=1, keepdim=True)
                x -= com
            xs[idx, ...] = x.detach().cpu().numpy()
            idx += 1

        # Adapt dt during burn-in
        if i < burn_in and i % save_every == 0:
            acc_rate = acc / (i + 1)
            if adaptive:
                if acc_rate < 0.3:
                    dt *= 1 - 0.5 * (0.3 - acc_rate.detach().cpu().item())
                elif acc_rate > 0.5:
                    dt *= 1 + 0.5 * (acc_rate.detach().cpu().item() - 0.5)

        if i % save_every == 0:
            print(f"Step {i}, acc_rate {acc / (i + 1):.3f}, dt {dt:.6f}")

    return xs, acc / n_steps


def float_or_none(value):
    if value.lower() == 'none':
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} must be a floating point number or 'None'")


def parse_args():
    params = argparse.ArgumentParser(description='parser example')
    params.add_argument('--logname', type=str, default='lj')
    params.add_argument('--run', type=int, default=0)
    params.add_argument('--save_every', type=int, default=100)
    params.add_argument('--nparticles', type=int, default=13)
    params.add_argument('--dim', type=int, default=3)
    params.add_argument('--kT', type=float, default=0.5)
    params.add_argument('--boxlength', type=float_or_none, default="None")
    params.add_argument('--sampler', choices=['MH', 'MALA'], default='MH')

    params.add_argument('--step_size', type=float, default=0.001)
    params.add_argument('--num_steps', type=int, default=20000)
    params.add_argument('--burn_in', type=int, default=10000)
    params.add_argument('--batch_size', type=int, default=10)
    params.add_argument('--spring_constant', type=float, default=0.5)
    params.add_argument('--adaptive_step_size', action='store_true', default=False)
    params.add_argument('--periodic', action='store_true', default=False)
    return params.parse_args()


if __name__ == "__main__":
    args = parse_args()

    boxlength = args.boxlength if args.boxlength is not None else args.nparticles ** (1 / args.dim)

    sys = LJ(
        nparticles=args.nparticles,
        dim=args.dim,
        batch_size=args.batch_size,
        device="cuda",
        boxlength=boxlength,
        kT=args.kT,
        epsilon=1.,
        sigma=2 ** (-1 / 6),
        cutoff=3,
        shift=False,
        periodic=args.periodic,
        spring_constant=args.spring_constant,
    )

    print(f"Boxlength: {boxlength}")

    x_init = torch.tensor(even_spacing(args.nparticles, boxlength, args.dim)).to("cuda")
    x_init = x_init.unsqueeze(0).expand(args.batch_size, -1, -1)
    x_init = x_init + torch.rand_like(x_init) * torch.sqrt(torch.tensor(2 * args.kT * 0.001))

    save_shape = (args.num_steps // args.save_every, args.batch_size, args.nparticles, args.dim)

    # Open an HDF5 file for writing
    with h5py.File(f'{args.logname}.h5', 'w') as h5file:
        traj_dataset = h5file.create_dataset('trajectory', shape=save_shape, dtype='f4')

        # Run the sampler, passing the HDF5 dataset for writing
        if args.sampler == 'MH':
            traj_dataset, acc = run_MH(sys, x_init, args.num_steps, args.step_size, args.adaptive_step_size,
                                       args.save_every, args.burn_in, traj_dataset)
        elif args.sampler == 'MALA':
            traj_dataset, acc = run_MALA(sys, x_init, args.num_steps, args.step_size, args.adaptive_step_size,
                                         args.save_every, args.burn_in, traj_dataset, center_com=True)
    # Now reopen the same file in read/write mode to reshape the dataset
    with h5py.File(f'{args.logname}.h5', 'r+') as h5file:
        traj_data = h5file['trajectory']
        frames, batch, particles, dim = traj_data.shape
        reshaped_shape = (frames * batch, particles, dim)

        # Read all data into memory
        print(f"Reshaping from {traj_data.shape} to {reshaped_shape}")
        full_data = traj_data[:]  # shape: (frames, batch, particles, dim)
        full_data = full_data.reshape(reshaped_shape)

        # Delete the old dataset
        del h5file['trajectory']

        # Create a new dataset with the reshaped data
        h5file.create_dataset('trajectory', data=full_data, dtype='f4')

    print("Reshaped and overwritten trajectory dataset.")

