import argparse
import numpy as np
import torch
from numpy.lib.format import open_memmap

from lj import LJ
from sampling_13 import run_MALA, run_MH, even_spacing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample 2D LJ16 system in a 4x4 box with PBC"
    )
    parser.add_argument('--sampler', choices=['MH', 'MALA'], default='MALA')
    parser.add_argument('--step_size', type=float, default=0.001)
    parser.add_argument('--num_steps', type=int, default=20000)
    parser.add_argument('--burn_in', type=int, default=20000)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--kT', type=float, default=0.5)
    parser.add_argument('--logname', type=str, default='lj16_2d')
    parser.add_argument('--adaptive_step_size', action='store_true', default=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    nparticles = 16
    dim = 2
    boxlength = 4.0

    system = LJ(
        nparticles=nparticles,
        dim=dim,
        batch_size=args.batch_size,
        device='cuda',
        boxlength=boxlength,
        kT=args.kT,
        epsilon=1.,
        sigma=2 ** (-1 / 6),
        cutoff=3,
        shift=False,
        periodic=True,
        spring_constant=0.5,
    )

    x_init = torch.tensor(even_spacing(nparticles, boxlength, dim)).to('cuda')
    x_init = x_init.unsqueeze(0).expand(args.batch_size, -1, -1)
    x_init = x_init + torch.rand_like(x_init) * torch.sqrt(torch.tensor(2 * args.kT * 0.001))

    shape = (
        args.num_steps // args.save_every,
        args.batch_size,
        nparticles,
        dim,
    )
    traj = open_memmap(
        f'{args.logname}_{args.kT}_temp.npy',
        dtype='float32',
        mode='w+',
        shape=shape,
    )

    if args.sampler == 'MH':
        traj, acc = run_MH(
            system,
            x_init,
            args.num_steps,
            args.step_size,
            args.adaptive_step_size,
            args.save_every,
            args.burn_in,
            traj,
        )
    else:
        traj, acc = run_MALA(
            system,
            x_init,
            args.num_steps,
            args.step_size,
            args.adaptive_step_size,
            args.save_every,
            args.burn_in,
            traj,
            center_com=False,
        )

    final_traj = open_memmap(
        f'{args.logname}_{args.kT}.npy',
        mode='w+',
        dtype=traj.dtype,
        shape=traj.shape,
    )
    final_traj[:] = traj[:]
    print(acc)
