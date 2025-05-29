import torch, argparse, os
import numpy as np
from lj import LJ
import matplotlib.pyplot as plt
from numpy.lib.format import open_memmap

def plot_distribution(x,name="x"):
    fig, ax = plt.subplots()
    print(x.shape)
    ax.hist(x.flatten(),bins=50, density=True)
    plt.show()
    ax.set_xlabel(name)
    ax.set_ylabel("Density")
    fig.savefig("%s.png"%name)

class BreakAllLoops(Exception):
    pass

def create_fcc_lattice(num_particles, boxlength):
    num_particles_original = num_particles
    if num_particles % 4 != 0:
        num_particles += 4 - (num_particles % 4)

    # Number of unit cells needed
    num_cells = num_particles // 4

    # Find the smallest cube number that fits all unit cells
    cells_per_side = int(np.ceil(num_cells ** (1. / 3.)))


    # Create an array to hold particle positions
    positions = np.zeros((num_particles_original, 3))

    # Index for the particle being placed
    particle = 0
    try:
        # Loop over each unit cell in the lattice
        for x in range(cells_per_side):
            for y in range(cells_per_side):
                for z in range(cells_per_side):
                    lattice_vector = [[0.0, 0.0, 0.0],
                                    [0.5, 0.5, 0.0],
                                    [0.5, 0.0, 0.5],
                                    [0.0, 0.5, 0.5]]
                    for vec in lattice_vector:
                        # Add the vector to the current cell
                        x_vec = np.array([x, y, z]) + vec
                        x_vec = x_vec
                        positions[particle] = x_vec
                        particle += 1
                        if particle == num_particles_original:
                            raise BreakAllLoops()
    except BreakAllLoops:
        pass
    # Scale by lattice constant
    positions *= boxlength/cells_per_side

    return positions

def even_spacing(nparticles, boxlength, dim):
    '''
    nparticles: number of particles
    boxlength: boxlength
    '''
    positions = np.zeros((nparticles, dim))
    num_per_dim = int(np.ceil(nparticles ** (1. / dim)))
    spacing = boxlength / num_per_dim
    try:
        if dim == 3:
            for i in range(num_per_dim):
                for j in range(num_per_dim):
                    for k in range(num_per_dim):
                        idx = i * num_per_dim ** 2 + j * num_per_dim + k
                        if idx >= nparticles:
                            raise BreakAllLoops()
                        positions[idx] = np.array([i, j, k]) * spacing
        elif dim == 2:
            for i in range(num_per_dim):
                for j in range(num_per_dim):
                    idx = i * num_per_dim + j
                    if idx >= nparticles:
                        raise BreakAllLoops()
                    positions[idx] = np.array([i, j]) * spacing

    except BreakAllLoops:
        pass
    return positions

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
    params.add_argument('--nparticles', type=int, default=38)
    params.add_argument('--dim', type=int, default=3)
    params.add_argument('--kT', type=float, default=0.18)
    params.add_argument('--boxlength', type=float_or_none, default="None")
    params.add_argument('--sampler', choices=['MH', 'MALA'], default='MALA')

    params.add_argument('--step_size', type=float, default=0.001)
    params.add_argument('--num_steps', type=int, default=50000)
    params.add_argument('--burn_in', type=int, default=10000)
    params.add_argument('--batch_size', type=int, default=2000)
    params.add_argument('--adaptive_step_size', action='store_true', default=False)
    params = params.parse_args()
    return params

def run_MH(target, x_init, n_steps, dt, adaptive=True, save_every=100, burn_in=1000, xs=None):
    '''
    MH sampler with burn-in, adaptive dt, and memory-mapped storage.
    
    target: has .potential(x) and .kT
    x_init: initial tensor of shape (batch, N, dim)
    xs: np.memmap array for saving samples (must be preallocated)
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

        # Burn-in plot data
        if i < burn_in:
            pot_list.append(potential_x.mean().item())

        # Save to memmap after burn-in
        if i >= burn_in and (i - burn_in) % save_every == 0:
            xs[idx] = x.detach().cpu().numpy()
            idx += 1

        # Adjust dt during burn-in
        if i < burn_in and i % save_every == 0:
            acc_rate = acc / (i + 1)
            if adaptive:
                if acc_rate < 0.3:
                    dt *= 1 - 0.5 * (0.3 - acc_rate.item())
                elif acc_rate > 0.5:
                    dt *= 1 + 0.5 * (acc_rate.item() - 0.5)
            print(f"Step {i}, acc_rate {acc_rate:.3f}, dt {dt:.6f}")

        # Burn-in plot
        if i == burn_in:
            pot_list = torch.tensor(pot_list) / 500
            plt.plot(pot_list.numpy())
            plt.savefig("pot_burn_in_mh.png")
            plt.close()

    return xs, acc / n_steps



def run_MALA(target, x_init, n_steps, dt, adaptive=True,
             save_every=100,burn_in=1000, xs=None):
    '''
    target: target with the potentiel we will run the langevin on
    -> needs to have force function implemented
    x (tensor): init points for the chains to update (batch_dim, dim)
    dt -> is multiplied by N for the phiFour before being passed
    '''

    acc = 0
    idx = 0
    pot_list = []
    force_current = target.force(x_init)
    potential_current = target.potential(x_init)
    x = x_init
    for i in range(n_steps+burn_in):
        x_new = x + dt * force_current
        x_new = x_new + np.sqrt(2 * dt * target.kT) * torch.randn_like(x)
        potential_new = target.potential(x_new)
        ratio = potential_current - potential_new
        if i<burn_in:
            pot_list.append(potential_current[0].clone())
        if i==burn_in:
            pot_list = torch.stack(pot_list)
            pot_list = pot_list
            plt.plot(pot_list.detach().cpu().numpy())
            #set y limits
            plt.ylim(pot_list.min().item(), 0)
            plt.savefig("pot_burn_in.png")
            plt.close()
        force_new = target.force(x_new)
        ratio -= ((x - x_new - dt * force_new) ** 2 / (4 * dt)).sum((1,2))
        ratio += ((x_new - x - dt * force_current) ** 2 / (4 * dt)).sum((1,2))
        ratio = 1/target.kT * ratio
        ratio = torch.exp(ratio)
        u = torch.rand_like(ratio)
        acc_i = u < torch.min(ratio, torch.ones_like(ratio))
        x[acc_i] = x_new[acc_i]
        force_current[acc_i] = force_new[acc_i]
        potential_current[acc_i] = potential_new[acc_i]
        acc += acc_i.float().mean()
        if i >= burn_in and (i-burn_in) % save_every == 0:
            xs[idx] = x.detach().cpu().numpy()
            idx += 1
        if (i < burn_in) and (i % save_every) == 0:
            acc_rate = acc/(i+1)
            if adaptive:
                if acc_rate <0.3:
                    dt *= 1 - 0.5*(0.3-acc_rate.detach().cpu().numpy())
                elif acc_rate >0.5:
                    dt *= 1 + 0.5*(acc_rate.detach().cpu().numpy()-0.5)
            print("Step %d, acc_rate %.3f, dt %.6f"%(i, acc_rate, dt))
    return xs, acc/n_steps


if __name__=="__main__":
    args = parse_args()
    periodic = False if args.boxlength is None else True
    sys = LJ(nparticles=args.nparticles, dim=args.dim, batch_size=args.batch_size,
                  device="cuda", boxlength=args.boxlength, kT=args.kT,
                  epsilon=1., sigma=1., cutoff=3, shift=False,
                  periodic=periodic)
    #initialize the particles on a lattice
    #x_init = torch.tensor(create_fcc_lattice(args.nparticles, args.boxlength)).to("cuda:0")
    if args.boxlength is None:
        boxlength = 3.0
    else:
        boxlength = args.boxlength
    x_init = torch.tensor(even_spacing(args.nparticles, boxlength, args.dim)).to("cuda:0")
    #x_init = torch.tensor(even_spacing(args.nparticles, args.boxlength, args.dim)).to("cpu")
    x_init = x_init.unsqueeze(0).expand(args.batch_size,-1,-1)
    x_init = x_init + torch.rand_like(x_init)*torch.sqrt(torch.tensor(2*args.kT*0.001))
    shape = (args.num_steps//args.save_every,args.batch_size,args.nparticles,args.dim)
    print(args.adaptive_step_size)
    traj_list = np.memmap(f'{args.logname}_{args.kT}_temp.npy', dtype='float32', mode='w+', shape=shape)
    if args.sampler == 'MH':
        traj_list, acc = run_MH(sys, x_init, args.num_steps, args.step_size, args.adaptive_step_size,
                                args.save_every, args.burn_in, traj_list)
    elif args.sampler == 'MALA':
        traj_list, acc = run_MALA(sys, x_init, args.num_steps, args.step_size, args.adaptive_step_size,
                                   args.save_every, args.burn_in, traj_list)
    y = open_memmap(f'{args.logname}_{args.kT}.npy', mode='w+', dtype=traj_list.dtype, shape=traj_list.shape)
    y[:] = traj_list[:]
    print(acc)
