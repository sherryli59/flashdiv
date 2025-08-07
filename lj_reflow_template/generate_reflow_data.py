import torch
import numpy as np
import os
from einops import rearrange, repeat, reduce
import argparse
import h5py

from flashdiv.flows.egnn_cutoff import EGNN_dynamicsPeriodic
from flashdiv.flows.egnn_periodic import EGNN_dynamicsPeriodic as EGNN_dynamicsPeriodic_noe


from flashdiv.flows.flow_net_torchdiffeq import FlowNet
# from flashdiv.flows.message_passing import
from flashdiv.flows.trainer import FlowTrainer, FlowTrainerTorus

from pytorch_lightning import Trainer, LightningModule
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.nn.functional as F
from pytorch_lightning import seed_everything
seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from flashdiv.lj.lj import LJ


def args_to_str(args, ignore=("ckpt_dir", "ckpt_name", "nparticles", "dim", "kT")):
    return "_".join([f"{k}_{v}" for k, v in vars(args).items() if k not in ignore])

def parse_args():
    parser = argparse.ArgumentParser(description="Flow sampling and relaxation")
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--nb_epochs', type=int, default=30)
    parser.add_argument('--init', type=str, default='uniform', help='Initialization method: normal, uniform')
    parser.add_argument('--nn', type=str, default='egnn')
    parser.add_argument('--ckpt_dir', type=str, default=None)  # optional override
    parser.add_argument('--nparticles', type=int, default=16)
    parser.add_argument('--kT', type=float, default=1.0)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--reflow', action='store_true', help='Use reflow data')
    return parser.parse_args()

def load_model(args):
    args_as_str = args_to_str(args)
    ckpt_path = args.ckpt_dir or f"flow_model_{args_as_str}/checkpoints/last.ckpt"
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"

    boxlength = args.nparticles ** (1 / args.dim)
    ljsystem = LJ(
        nparticles=args.nparticles,
        dim=args.dim,
        device=device,
        boxlength=boxlength,
        kT=args.kT,
        sigma=2 ** (-1 / 6),
        periodic=True
    )

    hidden_nf = 64
    nlayers = 3
    cutoff = 10.0
    max_neighbors = 14

    if args.nn == 'egnn_noe':
        net = EGNN_dynamicsPeriodic_noe(
            n_particles=ljsystem.nparticles,
            n_dimension=args.dim,
            device=device,
            hidden_nf=hidden_nf,
            boxlength=ljsystem.boxlength,
            act_fn=torch.nn.SiLU(),
            n_layers=nlayers,
            recurrent=True,
            tanh=True,
            attention=True,
            agg='sum')
    elif args.nn == 'egnn':
        net = EGNN_dynamicsPeriodic(
            n_particles=ljsystem.nparticles - 1,
            n_dimension=args.dim,
            device=device,
            hidden_nf=hidden_nf,
            act_fn=torch.nn.SiLU(),
            n_layers=nlayers,
            recurrent=True,
            tanh=True,
            attention=True,
            condition_time=True,
            out_node_nf=8,
            mode='egnn_dynamics',
            agg='sum',
            cutoff=cutoff,
            boxlength=ljsystem.boxlength,
            max_neighbors=max_neighbors
        )
    elif args.nn == 'egnn_lj':
        net = EGNN_dynamicsPeriodic(
        n_particles= ljsystem.nparticles - 1,
        device='cuda',
        n_dimension=ljsystem.dim,
        hidden_nf=hidden_nf,
        act_fn=torch.nn.SiLU(),
        n_layers=nlayers,
        recurrent=True,
        tanh=True,
        attention=True,
        condition_time=True,
        out_node_nf=8,
        mode='egnn_dynamics',
        agg='sum',
        cutoff=cutoff,
        boxlength=ljsystem.boxlength,
        max_neighbors=max_neighbors,
        distribution=ljsystem
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.nn}")

    trainer = FlowTrainerTorus.load_from_checkpoint(ckpt_path, flow_model=net,strict=False)
    return trainer.flow_model.eval(), ljsystem


def generate_samples(model, ljsystem, std=0.5, batch_size=1000, n_iterations=200):
    xt = []
    x0_list = []
    traj_list = []
    log_prob_list = []
    times = torch.linspace(0,1,100).to(device)
    for k in range(n_iterations):
        if args.init == 'normal':
            x0_, log_prob0 = ljsystem.sample_wrapped_gaussian(std = std, size = batch_size, device=device)
        elif args.init == 'uniform':
            x0_, log_prob0 = ljsystem.sample_uniform(size=batch_size, device=device)
        # permute randomly
        perm = torch.stack([torch.randperm(ljsystem.nparticles) for _ in range(x0_.shape[0])])
        idx = torch.arange(x0_.shape[0]).unsqueeze(-1).expand(x0_.shape[0], ljsystem.nparticles)
        x0 = x0_[idx, perm, :]
        with torch.no_grad():
            xt_, log_prob_ = model.sample_logprob(
                x0,
                log_prob0,
                times,
                method='rk4',
                options = {
                    'step_size': 1 / 100,
                    }
                )
            traj_ = xt_
            xt_ = xt_[-1]
        xt.append(xt_.detach())
        x0_list.append(x0.detach())
        traj_list.append(traj_.detach())
        log_prob_list.append(log_prob_[-1].detach())
    xt = rearrange(xt, 'l b p d -> (l b) p d')
    x0_list = rearrange(x0_list, 'l b p d -> (l b) p d')
    traj_list = rearrange(traj_list, 'l t b p d -> (l b) t p d')
    log_prob_list = rearrange(log_prob_list, 'l b -> (l b)')
    return xt, x0_list, log_prob_list, traj_list

def run_MALA(target, x_init, n_steps, dt, adaptive=True,
             save_every=100,burn_in=0, xs=None, center_com=False):
    '''
    target: target with the potentiel we will run the langevin on
    -> needs to have force function implemented
    x (tensor): init points for the chains to update (batch_dim, dim)
    dt -> is multiplied by N for the phiFour before being passed
    '''
    print("Running MALA")
    acc = 0
    idx = 0
    pot_list = []
    if xs is None:
        xs = torch.zeros((n_steps//save_every, x_init.shape[0], x_init.shape[1], x_init.shape[2]), device=x_init.device)
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
            if center_com:
                # Center the center of mass
                com = x.mean(dim=1, keepdim=True)
                x -= com
            xs[idx] = x
            idx += 1
        if (i < burn_in) and (i % save_every) == 0:
            acc_rate = acc/(i+1)
            if adaptive:
                if acc_rate <0.3:
                    dt *= 1 - 0.5*(0.3-acc_rate.detach().cpu().numpy())
                elif acc_rate >0.5:
                    dt *= 1 + 0.5*(acc_rate.detach().cpu().numpy()-0.5)
        if (i % save_every) == 0:
            print("Step %d, dt %.6f"%(i, dt))
    return xs

def relax_data(ljsystem, xt, batch_size=50, n_steps=100, dt=0.001, burn_in=30, save_every=5):
    """
    Relax the data using MALA.
    """
    print("Relaxing data")
    N, p, d = xt.shape
    xt_relaxed = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xt_batch = xt[start:end]  # shape: (B, p, d)
        # Run MALA: returns a list of saved states
        chain = run_MALA(
            ljsystem,
            xt_batch,
            n_steps=n_steps,
            dt=dt,
            burn_in=burn_in,
            save_every=save_every,
            adaptive=False,
        )
        # Take the last saved state as final sample
        xt_relaxed.append(chain[-1].detach())

    xt_relaxed = torch.cat(xt_relaxed, dim=0)
    return xt_relaxed


def load_h5_to_tensor(h5_fname, dataset_name='trajectory', device='cuda'):
    with h5py.File(h5_fname, 'r') as h5f:
        data_np = h5f[dataset_name][:]
    data_tensor = torch.from_numpy(data_np).to(device)
    return data_tensor
      
if __name__ == "__main__":
    args = parse_args()
    model, ljsystem = load_model(args)
    xt, x0_list, log_prob_list,traj_list = generate_samples(model, ljsystem, batch_size=2000, n_iterations=1)
    results = {
        'xt': xt,                          # Tensor or array
        'x0': x0_list,                  # List or Tensor
        'traj': traj_list,                # List or Tensor
        'log_prob': log_prob_list,      # List or Tensor
    }
    model_dir = f"flow_model_{args_to_str(args)}"
    os.makedirs(model_dir, exist_ok=True)
    reflow_data_path = os.path.join(model_dir, "reflow_data.pt")
    torch.save(results, reflow_data_path)
    # reflow_data = torch.load("data2/reflow_data.pt")

    # xt = reflow_data['xt']
    # x0_list = reflow_data['x0']
    # traj_list = reflow_data['traj']
    # xt_relaxed = relax_data(ljsystem, xt, batch_size=1000, n_steps=10, dt=0.0001, burn_in=0, save_every=5)
    # displacement = xt_relaxed - xt
    # boxlength = ljsystem.boxlength
    # displacement = displacement - torch.round(displacement / boxlength) * boxlength
    # displacement_norm = torch.norm(displacement, dim=-1)
    # potential = ljsystem.potential(xt_relaxed, turn_off_harmonic=True)
    # filter = potential < 0
    # print("Filtered out", xt_relaxed.shape[0] - filter.sum().item(), "samples")
    # xt_relaxed = xt_relaxed[filter]
    # x0_list = x0_list[filter]
    # results_relaxed = {
    #     "x0": x0_list,                # List or Tensor
    #     'xt': xt_relaxed,           # Relaxed tensor
    # }
    # torch.save(results_relaxed, 'data/reflow_data_relaxed.pt')


    # Step 1: Load simulation data
    sim_data = load_h5_to_tensor('lj162d_mala_periodic_1.0.h5', dataset_name='trajectory', device=device)
    sim_data = sim_data[::1000] 

    # Step 2: Evaluate logprob via reverse flow
    with torch.no_grad():
        x0, logprob_sim = model.sample_logprob(
            sim_data,
            logprob=None,        # default to 0
            reverse=True,
            method='rk4',
            options={'step_size': 1/100},
            boxlength=ljsystem.boxlength
        )
    # Step 3: Save or print
    print("Logprob of simulation data:", logprob_sim.mean().item())

    # Optional: save results
    torch.save({
        "sim_data": sim_data,
        "logprob": logprob_sim,
        "x0": x0
    }, os.path.join(model_dir, "simulation_logprob.pt"))
