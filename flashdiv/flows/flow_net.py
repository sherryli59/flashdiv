import torch.nn as nn
import torch
from einops import rearrange, repeat, reduce
from torch.func import jvp

# base class
class FlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = lambda x, t: torch.cat([x, t], dim=1)

    def forward(self, x, t):
        raise NotImplementedError("Override this method in subclasses")

    @torch.no_grad()
    def divergence2(self, x,t, div_samples=int(1e3)):
        """
        hutchison trace estimator
        """
        x0 = repeat(x, 'b p d -> (b r) p d', r=div_samples)
        t = repeat(t, 'b -> (b r)', r=div_samples)
        v = torch.randn_like(x0)

        ouptut, jacvp = jvp(lambda x : self.forward(x, t),
        (x0,),
        (v,),
        )
        # print(jacvp.shape)

        tr = reduce(
            reduce(
                rearrange(
                    v * jacvp,
                    '(b r) p d -> b r p d',
                    r=div_samples),
                'b r p d -> b r',
                'sum'),
            'b r  -> b ',
            'mean')

        # takes care of residual
        torch.cuda.empty_cache()

        return tr

    @torch.no_grad()
    def sample(self, xs, n_steps: int=100):
        """
        ODE integration returning only the final state
        """
        dt = 1. / n_steps
        xs = xs.detach().clone()
        batch_size = xs.shape[0]
        for i in range(n_steps):
            t = torch.ones(batch_size).to(xs) * i * dt
            vt = self.forward(xs, t)
            xs = xs.detach().clone() + dt * vt
        return xs


    @torch.no_grad()
    def sample_traj(self, xs, n_steps: int=100):
        """
        ODE integration returning the trajectory
        """
        dt = 1. / n_steps
        xs = xs.detach().clone()
        all_xs = [xs]
        all_ts = [0.0]
        batch_size = xs.shape[0]
        for i in range(n_steps):
            t = torch.ones(batch_size).to(xs) * i * dt
            vt = self.forward(xs, t)
            xs = xs.detach().clone() + dt * vt
            all_xs.append(xs)
            all_ts.append((i+1) * dt)
        return torch.tensor(all_ts).to(xs), torch.stack(all_xs).to(xs)

    @torch.no_grad()
    def sample_traj_logprob(self, xs, n_steps: int=100, **kwargs):
        """
        ODE integration returning the trajectory and logprob
        """
        dt = 1. / n_steps
        xs = xs.detach().clone()
        all_xs = [xs]
        all_ts = [0.0]
        curr_trace = torch.zeros((xs.shape[0])).to(xs)
        all_traces = [curr_trace]
        batch_size = xs.shape[0]
        for i in range(n_steps):
            t = torch.ones(batch_size).to(xs) * i * dt
            curr_trace += self.divergence(xs, t, **kwargs) * dt
            all_traces.append(curr_trace)
            vt = self.forward(xs, t)
            xs = xs.detach().clone() + dt * vt
            all_xs.append(xs)
            all_ts.append((i+1) * dt)
        return torch.tensor(all_ts), torch.stack(all_xs), rearrange(all_traces, 't b  -> t b ')

    def sample_logprob(self, xs, n_steps: int=100, **kwargs):
        """
        ODE integration returning the last position and associated logprob
        """
        dt = 1. / n_steps
        xs = xs.detach().clone()
        curr_trace = torch.zeros((xs.shape[0])).to(xs)
        curr_trace.requires_grad_(False)
        batch_size = xs.shape[0]
        for i in range(n_steps):
            t = torch.ones(batch_size).to(xs) * i * dt
            xs = xs.detach().clone()
            xs.requires_grad_(True)
            with torch.enable_grad():
                div = self.divergence(xs, t, **kwargs).detach()
                curr_trace = curr_trace + div * dt
            vt = self.forward(xs, t)
            xs = xs + dt * vt
        return xs, curr_trace
