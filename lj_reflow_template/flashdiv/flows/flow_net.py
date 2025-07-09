import torch.nn as nn
import torch
from einops import rearrange, repeat, reduce
from torch.func import jvp, vmap, jacrev
from torchdiffeq import odeint as odeint



class FlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = lambda x, t: torch.cat([x, t], dim=1)

    def forward(self, x, t):
        raise NotImplementedError("Override this method in subclasses")

    @torch.no_grad()
    def divergence_hutch(self, x,t, div_samples=int(1e3), **kwargs):
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
    def divergence_full_jacobian(self, x,t, **kwargs):
        """
        Computes the full jacobian and then selects the diagonal
        """

        jac = jacrev(
            lambda x, t : self.forward(x.unsqueeze(0), t.unsqueeze(0)).squeeze(0),
            argnums=0
        )

        vmapped_jac = vmap(jac, in_dims=(0, 0))

        batched_jacobian = vmapped_jac(x, t) #(b p d p d)

        return torch.einsum(
            'b p d p d -> b',
            batched_jacobian
        )

    @torch.no_grad()
    def direct_trace(self, x,t, **kwargs):
        """
        Computes the full jacobian and then selects the diagonal
        """
        def f(x):
            return self.forward(x, t)
        shape = x.shape
        def _func_sum(x):
            return f(x.reshape(shape)).sum(dim=0).flatten()
        jacobian = torch.autograd.functional.jacobian(_func_sum, x.reshape(x.shape[0],-1), create_graph=False).transpose(0,1)
        return torch.vmap(torch.trace)(jacobian).flatten()

    @torch.no_grad()
    def sample(self, x0, times,**kwargs):
        """
        input : x0 (batch_size, napart, dim)
        times : (n_steps, ) evaluations times

        the kwargs should corresponf to those of the odeint function
        """
        batch_size = x0.shape[0]
        npart = x0.shape[-2]
        dim = x0.shape[-1]

        if 'method' not in  kwargs:
            kwargs['method'] = 'euler'
        # print(kwargs)

        # little reshaping here
        def integration_func(t, xs):
            t_ = torch.ones(batch_size).to(xs) * t.item()
            return self.forward(xs, t_).detach()

        return odeint(integration_func, x0, times, **kwargs)


    # @torch.no_grad()
    def sample_logprob(self, x0, logprob0, times, verbose=False,**kwargs):
        """
        ODE integration returning the trajectory and logprob
        """
        batch_size = x0.shape[0]
        npart = x0.shape[-2]
        dim = x0.shape[-1]

        if 'method' not in  kwargs:
            kwargs['method'] = 'euler'
        if 'options' not in kwargs:
            kwargs['options'] = {'step_size': 1 / 100}

        # some logic to determine which divergence to use.
        div_kwargs = {}
        if 'div_method' in kwargs:
                if kwargs['div_method'] == 'hutch':
                    self._divergence = self.divergence_hutch
                    if 'div_samples' in kwargs:
                        div_kwargs['div_samples'] = kwargs.pop('div_samples')
                elif kwargs['div_method'] == 'full_jacobian':
                    self._divergence = self.divergence_full_jacobian
                elif kwargs['div_method'] == 'direct_trace':
                    self._divergence = self.direct_trace
                else:
                    raise ValueError(f"Unknown divergence method: {kwargs['div_method']}, possible values are 'hutch', 'full_jacobian, direct_trace'")
                del kwargs['div_method'] # because we pas to odeint after
        elif hasattr(self, 'divergence'):
            self._divergence = self.divergence
        else:
            print("No divergence method specified, using hutchison trace estimator by default")
            self._divergence = self.divergence_hutch

        if verbose:
            print("Using divergence method:", self._divergence.__name__)

        state0 = torch.cat(
            (x0,
            repeat(
                logprob0,
                'b -> b p d',
                p=npart, d=dim
            )),
            dim=0
        )

        # little reshaping here
        def integration_func(t, state):
            xs = state[:batch_size]
            t_ = torch.ones(batch_size).to(xs) * t.item()
            v = self.forward(xs, t_).detach()
            div = self._divergence(xs, t_, **div_kwargs).detach()
            return torch.cat(
                (v,
                repeat(
                    - div,
                    'b -> b p d',
                    p=npart, d=dim
                )),
                dim=0
            ).detach()

        integrated_state = odeint(integration_func, state0, times, **kwargs)
        all_xs = integrated_state[:, :batch_size]
        all_logprobs = integrated_state[:, batch_size:, 0, 0]

        return all_xs, all_logprobs











