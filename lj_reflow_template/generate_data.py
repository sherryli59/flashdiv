import os
import math
import torch
import numpy as np
from einops import rearrange
import argparse
from pytorch_lightning import seed_everything

from flashdiv.flows.egnn_cutoff import EGNN_dynamicsPeriodic
from flashdiv.flows.egnn_periodic import EGNN_dynamicsPeriodic as EGNN_dynamicsPeriodic_noe
from flashdiv.flows.trainer import FlowTrainerTorus
from flashdiv.lj.lj import LJ


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def args_to_str(args, ignore=("ckpt_dir", "ckpt_name", "nparticles", "dim", "kT")):
    """Compact string representation of arguments for folder names."""
    return "_".join([f"{k}_{v}" for k, v in vars(args).items() if k not in ignore])


def parse_args():
    parser = argparse.ArgumentParser(description="Generate reflow or final data")
    parser.add_argument('--nn', type=str, default='egnn')
    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--nparticles', type=int, default=16)
    parser.add_argument('--kT', type=float, default=1.0)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--n_iterations', type=int, default=1,
                        help='Number of batches of samples to draw')
    parser.add_argument('--init', type=str, default='uniform',
                        help='Initial distribution for reflow data')
    parser.add_argument('--mode', choices=['reflow', 'final'], default='reflow',
                        help='Which type of data to generate')
    parser.add_argument('--reflow_data', type=str, default=None,
                        help='Existing reflow data (required for mode=final)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Where to store the generated data')
    return parser.parse_args()


def load_model(args):
    """Load a trained flow model and the associated LJ system."""
    boxlength = args.nparticles ** (1 / args.dim)
    ljsystem = LJ(
        nparticles=args.nparticles,
        dim=args.dim,
        device=device,
        boxlength=boxlength,
        kT=args.kT,
        sigma=2 ** (-1 / 6),
        periodic=True,
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
    else:
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
            max_neighbors=max_neighbors,
        )
    trainer = FlowTrainerTorus.load_from_checkpoint(args.ckpt_dir, flow_model=net, strict=False)
    return trainer.flow_model.eval(), ljsystem


def generate_samples(model, ljsystem, batch_size, n_iterations, init='uniform', x0_data=None):
    """Generate samples from the model.

    If ``x0_data`` is provided the model starts from these initial positions,
    otherwise fresh samples are drawn from the LJ system.
    """
    times = torch.linspace(0, 1, 100).to(device)
    xt_list, x0_list, traj_list, log_prob_list = [], [], [], []
    for k in range(n_iterations):
        if x0_data is None:
            if init == 'normal':
                x0_, log_prob0 = ljsystem.sample_wrapped_gaussian(std=0.5, size=batch_size, device=device)
            else:
                x0_, log_prob0 = ljsystem.sample_uniform(size=batch_size, device=device)
            perm = torch.stack([torch.randperm(ljsystem.nparticles) for _ in range(x0_.shape[0])])
            idx = torch.arange(x0_.shape[0]).unsqueeze(-1).expand(x0_.shape[0], ljsystem.nparticles)
            x0 = x0_[idx, perm, :]
        else:
            start = k * batch_size
            end = min(start + batch_size, x0_data.shape[0])
            x0 = x0_data[start:end].to(device)
            log_prob0 = torch.zeros(x0.shape[0], device=device)
        with torch.no_grad():
            xt_, log_prob_ = model.sample_logprob(
                x0,
                log_prob0,
                times,
                method='rk4',
                options={'step_size': 1 / 100},
            )
            traj_ = xt_
            xt_ = xt_[-1]
        xt_list.append(xt_.detach())
        x0_list.append(x0.detach())
        traj_list.append(traj_.detach())
        log_prob_list.append(log_prob_[-1].detach())
    xt = rearrange(xt_list, 'l b p d -> (l b) p d')
    x0_list = rearrange(x0_list, 'l b p d -> (l b) p d')
    traj_list = rearrange(traj_list, 'l t b p d -> (l b) t p d')
    log_prob_list = rearrange(log_prob_list, 'l b -> (l b)')
    return xt, x0_list, log_prob_list, traj_list


def main():
    args = parse_args()
    seed_everything(42)
    model, ljsystem = load_model(args)
    if args.mode == 'final':
        assert args.reflow_data is not None, 'reflow_data is required in final mode'
        reflow = torch.load(args.reflow_data)
        x0_ref = reflow['x0']
        n_iters = math.ceil(x0_ref.shape[0] / args.batch_size)
        xt, x0_list, log_prob_list, traj_list = generate_samples(
            model, ljsystem, args.batch_size, n_iters, x0_data=x0_ref
        )
        default_name = 'final_data.pt'
    else:
        xt, x0_list, log_prob_list, traj_list = generate_samples(
            model, ljsystem, args.batch_size, args.n_iterations, init=args.init
        )
        default_name = 'reflow_data.pt'
    results = {
        'xt': xt,
        'x0': x0_list,
        'log_prob': log_prob_list,
        'traj': traj_list,
    }
    if args.save_path is None:
        model_dir = f"flow_model_{args_to_str(args)}"
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, default_name)
    else:
        save_path = args.save_path
    torch.save(results, save_path)


if __name__ == '__main__':
    main()

