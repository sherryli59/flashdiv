import torch
import torch.nn as nn
import sys

from flashdiv.flows.trainer import FlowTrainer, DistillationTrainer
from flashdiv.lj.lj import LJ


import matplotlib.pyplot as plt
from einops import rearrange, repeat, reduce, einsum
import math

from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import numpy as np


import torch.nn.functional as F
from torch.func import jvp
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from flashdiv.flows.eqtf_com import EqTransformerFlowSherryVariation
from flashdiv.flows.trainer import FlowTrainer
from flashdiv.flows.egnn_et import EasyTrace_EGNN
from flashdiv.flows.etvf import EasyTraceVelocityField
from flashdiv.flows.eqtf_pair import EqTransformerFlow as EqTransformerFlow_pair
from flashdiv.flows.eqtf_gen import EqTransformerFlow as EqTransformerFlow
from flashdiv.flows.new_vf import EqTransformerFlow as NewEqTransformerFlow
from flashdiv.flows.egnn_new.egnn import EGNN_dynamics_Noe
from flashdiv.flows.egnn import EGNN_dynamics
from flashdiv.flows.egnn_new.egnn_var import EGNN_dynamics as EGNN_dynamics_var




from types import SimpleNamespace    # lightweight stand-in for argparse.Namespace


#plt.style.use('my_style')
#use default matplotlib style
plt.style.use('default')

# Example scripe

plt.rcParams.update({'text.usetex': True,
                     'font.family': 'CMU',
                     'text.latex.preamble': r'\usepackage{amsfonts}'})
nbparticles = 13
dim = 3

def parse_args(args):
    if args.nn == 'egnn':
        velocitynet = EGNN_dynamics(n_particles=nbparticles, n_dimension=dim, device=device, hidden_nf=int(args.gnn_hidden_dim),
        act_fn=torch.nn.SiLU(), n_layers=int(args.nb_layers), recurrent=True, tanh=True, attention=True, agg='sum')
    elif args.nn == 'egnn_et':
        velocitynet = EasyTrace_EGNN(n_particles=nbparticles, hidden_nf=int(args.gnn_hidden_dim),
            act_fn=torch.nn.SiLU(), n_layers=int(args.nb_layers), recurrent=True, tanh=True, attention=True, agg='sum')
    elif args.nn == "etvf":
        velocitynet = EasyTraceVelocityField(n_particles=nbparticles,gnn_hidden_dim = int(args.gnn_hidden_dim),
                                            tf_hidden_dim = int(args.tf_hidden_dim),act_fn=torch.nn.SiLU(), n_layers=int(args.nb_layers), 
                                            recurrent=True, tanh=True, attention=True, agg='sum')
    elif args.nn == "eqtf_sherry":
        velocitynet = EqTransformerFlowSherryVariation(
            input_dim=3,
            embed_dim=int(args.tf_hidden_dim))
    elif args.nn == "eqtf_pair":
        velocitynet = EqTransformerFlow_pair(
            n_particles=nbparticles,
            hidden_nf=int(args.tf_hidden_dim))
    elif args.nn == "eqtf":
        velocitynet = EqTransformerFlow(
        n_particles=nbparticles,
        hidden_nf=int(args.tf_hidden_dim))
    elif args.nn == "new_vf":
        velocitynet = NewEqTransformerFlow(
        n_particles=nbparticles,
        hidden_nf=int(args.tf_hidden_dim),
        gnn_hidden_nf=int(args.gnn_hidden_dim),
        )
    elif args.nn == "egnn_noe":
        velocitynet = EGNN_dynamics_Noe(
        n_particles=nbparticles - 1,
        device=device,
        n_dimension=dim,
        hidden_nf=12,
        act_fn=torch.nn.SiLU(),
        n_layers=2,
        recurrent=True,
        tanh=True,
        attention=True,
        condition_time=True,
        # in_node_nf=1,  # 1 for time, 2 for position
        out_node_nf=12, # expressivity for potential
        )
    elif args.nn == "egnn_var":
        velocitynet = EGNN_dynamics_var(
        n_particles=nbparticles,
        device=device,
        n_dimension=dim,
        hidden_nf=12,
        act_fn=torch.nn.SiLU(),
        n_layers=2,
        recurrent=True,
        tanh=True,
        attention=True,
        condition_time=True,
        # in_node_nf=1,  # 1 for time, 2 for position
        out_node_nf=12, # expressivity for potential
        )
    return velocitynet

def load_model(nn,ckpt_path):
    args = SimpleNamespace(
        nn=nn,            # pick one of: "egnn", "etvf", "eqtf_sherry", "eqtf"
        gnn_hidden_dim=32,
        tf_hidden_dim=256,
        nb_layers=4
    )

    velocitynet = parse_args(args)
    ckpt = torch.load(ckpt_path, map_location='cuda')

    # Assume `ckpt` is a raw state_dict, or nested inside a dict under "state_dict"
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    velocityTrainer = FlowTrainer(velocitynet)
    velocityTrainer.load_state_dict(state_dict)
    # # velociTrainer = FlowTrainer.load_from_checkpoint(flow_model = velocitynet) #
    velocitynet = velocityTrainer.flow_model.to(device)
    print(f"nb params : {sum(p.numel() for p in velocityTrainer.parameters())}")
    return velocitynet

def logprob0(x, scale):
    """
    x : (batch, p, dim)
    """
    return - reduce(x**2, 'b p d -> b', 'sum') / (2 * scale**2)

def generate_samples_and_prob(model, batch_size=50, n_iterations=200):
    xt = []
    target_log_prob = []
    x0_list = []
    times = torch.linspace(0,1,2).to(device)
    for k in range(n_iterations):
        # x0 = gen_sorted_gaussian(batch_size, nbparticles, dim, noise_scale = 0.5).to(device)
        x0 = torch.randn(batch_size, nbparticles, dim).to(device) * 1.0
        x0 -= x0.mean(dim=1, keepdim=True)  # center the particles
        with torch.no_grad():
            xt_, target_log_prob_ = model.sample_logprob(x0, logprob0(x0, 1.0), times, div_method='direct_trace', method='rk4',
            options = {
                'step_size': 1 / 30,
                # 'max_num_steps': int(1e3)
                }
            )
            xt_ = xt_[-1]
            target_log_prob_ = target_log_prob_[-1]
        xt.append(xt_.detach())
        target_log_prob.append(target_log_prob_.detach())
        x0_list.append(x0.detach())
    xt = rearrange(xt, 'l b p d -> (l b) p d')
    target_log_prob = rearrange(target_log_prob, 'l b -> (l b)')
    x0_list = rearrange(x0_list, 'l b p d -> (l b) p d')

    return xt, target_log_prob, x0_list

if __name__ == "__main__":
    # Example usage
    # nn_type="egnn_noe"
    # ckpt_path = "flow_model_learning_rate_0.0001_batch_size_256_nb_epochs_60_gnn_hidden_dim_32_tf_hidden_dim_256_temp_1.0_nb_layers_4_nn_egnn_noe/checkpoints/last.ckpt"
    # egnn_noe = load_model(nn_type,ckpt_path)
    # xt, target_log_prob, x0_list = generate_samples_and_prob(egnn_noe, batch_size=50, n_iterations=200)

    # # Pack results into a dictionary
    # results = {
    #     'xt': xt,                          # Tensor or array
    #     'target_log_prob': target_log_prob, # Tensor or array
    #     'x0_list': x0_list                  # List or Tensor
    # }

    # # Example: Save as a PyTorch file
    # torch.save(results, 'egnn_results.pt')

    nn_type="egnn_var"
    ckpt_path = "flow_model_learning_rate_0.0001_batch_size_256_nb_epochs_60_gnn_hidden_dim_32_tf_hidden_dim_256_temp_1.0_nb_layers_4_nn_egnn_var/checkpoints/last.ckpt"
    model = load_model(nn_type,ckpt_path)
    xt, target_log_prob, x0_list = generate_samples_and_prob(model, batch_size=50, n_iterations=200)
    results = {
        'xt': xt,                          # Tensor or array
        'target_log_prob': target_log_prob, # Tensor or array
        'x0_list': x0_list                  # List or Tensor
    }
    torch.save(results, 'egnn_var_results.pt')
