import torch
import torch.nn as nn
import numpy as np
import sys
from einops import rearrange, repeat, reduce

from flashdiv.flows.egnn_cutoff import EGNN_dynamics, EGNN_dynamicsPeriodic

from flashdiv.flows.flow_net_torchdiffeq import FlowNet
# from flashdiv.flows.message_passing import
from flashdiv.flows.trainer import FlowTrainer, FlowTrainerTorus

from pytorch_lightning import Trainer, LightningModule
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.nn.functional as F
from pytorch_lightning import seed_everything
seed_everything(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from flashdiv.lj.lj import LJ


def load_model(ckpt_path):


    hidden_nf = 64
    nlayers = 3
    cutoff = 10.0
    max_neighbors = 14


    velocitynet = EGNN_dynamicsPeriodic(
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
        max_neighbors=max_neighbors
        ).to(device)

    velocitytrainer = FlowTrainerTorus.load_from_checkpoint(
        ckpt_path,
        flow_model = velocitynet)
    velocitynet = velocitytrainer.flow_model.eval().to(device)
    return velocitynet



def generate_samples(model, ljsystem, std=0.5, batch_size=1000, n_iterations=200):
    xt = []
    x0_list = []
    traj_list = []
    log_prob_list = []
    times = torch.linspace(0,1,100).to(device)
    for k in range(n_iterations):
        #x0_, log_prob0 = ljsystem.sample_wrapped_gaussian(std = std, size = batch_size, device=device)
        x0, log_prob0 = x0_reflow[k*batch_size:(k+1)*batch_size].to(device), torch.zeros(batch_size).to(device)
        xt_ref = xt_reflow[k*batch_size:(k+1)*batch_size].to(device)
        with torch.no_grad():
            xt_, log_prob_ = model.sample_logprob(
                x0,
                log_prob0,
                times=times,
                method='rk4',
                options = {
                    'step_size': 1 / 100,
                    }
                )
            traj_ = xt_
            xt_ = xt_[-1]
            #compare with xt_ref
            print(xt_ref[0], xt_[0])
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


        
if __name__ == "__main__":
    nparticles = 16
    dim = 2
    ljsystem= LJ(
    nparticles=nparticles,
    dim=dim,
    device='cuda',
    boxlength= nparticles ** (1 / dim),
    sigma=2 ** (-1 / 6),
    # shift=False,
    kT=1.0,
    #   spring_constant = 0.05,
    periodic=True
    )
    std=0.5
    model = load_model("flow_model_learning_rate_0.0005_batch_size_256_nb_epochs_30_init_uniform_nn_egnn_reflow_True/checkpoints/last.ckpt")
    reflow_data = torch.load("data2/reflow_data_filtered.pt")
    x0_reflow = reflow_data['x0']  # shape: [N_total, p, d]
    xt_reflow = reflow_data['xt']
    xt, x0_list, log_prob_list,traj_list = generate_samples(model, ljsystem, batch_size=1000, n_iterations=1)
    results = {
        'xt': xt,                          # Tensor or array
        'x0': x0_list,                  # List or Tensor
        'log_prob': log_prob_list,       # List or Tensor
        'traj': traj_list                # List or Tensor
    }
    torch.save(results, 'data/final_data.pt')

