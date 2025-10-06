import torch.nn as nn
import torch
from einops import rearrange, repeat, reduce
from torch.func import jvp, jacrev, vmap
# import ode solver class
from torchdiffeq import odeint, odeint_adjoint

# base class
class FlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = lambda x, t: torch.cat([x, t], dim=1)

    def forward(self, x, t):
        raise NotImplementedError("Override this method in subclasses")


    def divergence_hutch(self, x, t, *, div_samples: int = 1,
                            differentiable: bool = False, noise: str = "rademacher"):
        """
        Estimate tr(J_f(x,t)) via Hutchinson using forward-mode AD:
            tr(J) = E_eps[ eps^T (J eps) ].
        - x: (B, P, D), t: (B,)
        - div_samples: # of probes (1–4 for training is typical)
        - differentiable: keep graph for backprop into model params
        - noise: 'rademacher' (±1) or 'gaussian'
        Returns: (B,) divergence estimate per sample.
        """
        B = x.shape[0]

        # We need grads wrt parameters (not x), so x doesn't need requires_grad.
        grad_ctx = torch.enable_grad() if differentiable else torch.no_grad()
        with grad_ctx:
            def f(xx):
                return self.forward(xx, t)  # preserves batch structure

            acc = torch.zeros(B, device=x.device, dtype=x.dtype)

            for _ in range(int(div_samples)):
                if noise == "rademacher":
                    eps = torch.empty_like(x).bernoulli_(0.5).mul_(2.).sub_(1.)  # ±1
                elif noise == "gaussian":
                    eps = torch.randn_like(x)
                else:
                    raise ValueError("noise must be 'rademacher' or 'gaussian'")

                # jvp returns (f(x), J_x f · eps); we only need the second
                _, jvp_out = jvp(f, (x,), (eps,))  # shapes match x

                acc += (jvp_out.reshape(B, -1) * eps.reshape(B, -1)).sum(dim=1)

            return acc / float(div_samples)


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

    def direct_trace(self, x,t, **kwargs):
        """
        Computes the full jacobian and then selects the diagonal
        """
        def f(x):
            return self.forward(x, t)
        shape = x.shape
        def _func_sum(x):
            return f(x.reshape(shape)).sum(dim=0).flatten()
        jacobian = torch.autograd.functional.jacobian(_func_sum, x.reshape(x.shape[0],-1), create_graph=kwargs.get("differentiable", True)).transpose(0,1)
        return torch.vmap(torch.trace)(jacobian).flatten()

    @torch.no_grad()
    def sample(self, x0, times, return_traj: bool = True, **kwargs):
        """
        input : x0 (batch_size, napart, dim)
        times : (n_steps, ) evaluations times

        the kwargs should corresponf to those of the odeint function
        """
        batch_size = x0.shape[0]
        npart = x0.shape[-2]
        dim = x0.shape[-1]
        verbose = kwargs.pop('verbose', False)
        if verbose:
            print(kwargs)
        boxlength = kwargs.pop('boxlength', None)

        if 'method' not in kwargs:
            kwargs['method'] = 'rk4'
        if 'options' not in kwargs:
            kwargs['options'] = {'step_size': 1 / 100}
        # print(kwargs)

        # little reshaping here
        # inorder to have some callback to the forward method we need to define this as a class

        # we do this so we can pass some callbacks
        class IntegrationFunc:
            def __init__(self, model):
                self.model = model

            def __call__(self, t, xs):
                # keep computation on-device without CPU sync
                t_ = t.to(xs).expand(batch_size)
                return self.model.forward(xs, t_)

        integration_func = IntegrationFunc(self)

        # watch out, I had to modify the core code for callback to act on the state
        if boxlength is not None:

            # this is an inplace modification of xs
            def mod(xs):
                xs[:] = (xs + 0.5 * boxlength) % boxlength - 0.5 * boxlength

            setattr(integration_func, 'callback_step', lambda t, xs, dt: mod(xs)) # this is an inplace operation on xs, which we carry on throught the next integration step

        trajectory = odeint(integration_func, x0, times, **kwargs)
        final_state = trajectory[-1]
        if return_traj:
            return final_state, trajectory
        return final_state, None

  
    def sample_logprob(self, x, logprob=None, times=None, reverse=False,
                            differentiable: bool=False, return_traj=True, **kwargs):
        """
        Integrate (x, logp) as a tuple. logp is (B,), not tiled.
        """
        device = x.device
        if times is None:
            times = torch.linspace(0., 1., 2, device=device)
        if reverse:
            times = torch.flip(times, dims=[0])

        div_kwargs = self._select_divergence_impl_(kwargs)

        # ODE options
        method = kwargs.pop('method', 'rk4')
        options = kwargs.pop('options', {'step_size': 1/100})

        B = x.shape[0]
        if logprob is None:
            if reverse:
                logprob = torch.zeros(B, device=device)
            else:
                raise ValueError("logprob must be provided for forward sampling.")

        grad_ctx = torch.enable_grad() if differentiable else torch.no_grad()
        with grad_ctx:
            def func(t, state):
                xs, lp = state                           # xs: (B,P,D), lp: (B,)
                t_ = t.to(xs).expand(B)
                if hasattr(self, 'forward_and_divergence'):
                    v, div = self.forward_and_divergence(xs, t_, differentiable=differentiable, **div_kwargs)
                else:
                    v = self.forward(xs, t_)                 # (B,P,D)
                    # cheap analytic divergence; MUST be graph-free if differentiable=False
                    div = self._divergence(xs, t_, differentiable=differentiable, **div_kwargs)  # (B,)
                dxdt   =  v
                dlpdt  = -div
                return (dxdt, dlpdt)

            if differentiable:
                y = odeint_adjoint(func, (x, logprob), times, method=method, options=options,
                                adjoint_params=tuple(self.parameters()))
            else:
                y = odeint(func, (x, logprob), times, method=method, options=options)

            xs_traj, lp_traj = y
            xs, lp = xs_traj[-1], lp_traj[-1]
            if return_traj:
                return xs, lp, xs_traj, lp_traj
            else:
                return xs, lp, None, None

    def _select_divergence_impl_(self, kwargs):
        # (shared with your sample_logprob) — returns callable self._divergence and div_kwargs
        div_kwargs = {}
        if hasattr(self, 'divergence'):
            print("Using custom divergence method")
            self._divergence = self.divergence
        elif kwargs.get('div_method') == 'hutch':
            self._divergence = self.divergence_hutch
            if 'div_samples' in kwargs:
                div_kwargs['div_samples'] = kwargs.pop('div_samples')
        elif kwargs.get('div_method') == 'full_jacobian':
            self._divergence = self.divergence_full_jacobian
        else:
            self._divergence = self.direct_trace
        kwargs.pop('div_method', None)
        return div_kwargs

    def integrate_augmented(self, x0, times, *, reverse=False,
                            differentiable: bool=False,
                            method: str='rk4', options=None, **kwargs):
        """
        Integrate augmented state (x, logq, s_div2, s_v2):
          dx/dt = v(x,t)
          dlogq/dt = - tr J_v(x,t)
          ds_div2/dt = (tr J_v)^2
          ds_v2/dt   = ||v||^2
        Returns:
          xT, logqT, int_div2, int_v2, (optionally trajs if return_traj=True)
        """
        if options is None:
            options = {'step_size': 1/100}
        device = x0.device
        if times is None:
            times = torch.linspace(0., 1., 2, device=device)
        if reverse:
            times = torch.flip(times, dims=[0])

        # choose divergence impl once
        div_kwargs = self._select_divergence_impl_(kwargs)

        B = x0.shape[0]
        # init logq (forward direction requires it; if you're only regularizing, zeros are fine too)
        logq0 = torch.zeros(B, device=device, dtype=x0.dtype)

        # augmented accumulators as zeros
        s0 = torch.zeros(B, device=device, dtype=x0.dtype)  # for div^2
        r0 = torch.zeros(B, device=device, dtype=x0.dtype)  # for ||v||^2

        grad_ctx = torch.enable_grad() if differentiable else torch.no_grad()
        with grad_ctx:
            def func(t, state):
                x, logq, s_div2, s_v2 = state                      # x: (B,P,D) or (B,D)
                t_ = t.to(x).expand(B)

                v  = self.forward(x, t_)                           # (B,*,D)
                dv = self._divergence(x, t_, differentiable=differentiable, **div_kwargs)  # (B,)

                # integrands
                dlpdt    = -dv
                ds_div2  = dv.pow(2)
                v_flat   = v.reshape(B, -1)
                ds_v2    = (v_flat * v_flat).sum(dim=1)

                if not differentiable:
                    v = v.detach(); dlpdt = dlpdt.detach()
                    ds_div2 = ds_div2.detach(); ds_v2 = ds_v2.detach()

                return (v, dlpdt, ds_div2, ds_v2)

            state0 = (x0, logq0, s0, r0)

            if differentiable:
                y = odeint_adjoint(func, state0, times, method=method, options=options,
                                   adjoint_params=tuple(self.parameters()))
            else:
                y = odeint(func, state0, times, method=method, options=options)

            xT, logqT, s_div2_T, s_v2_T = (y_i[-1] for y_i in y)
            return xT, logqT, s_div2_T, s_v2_T

    
    def estimate_logprob(self, x, base_dist, return_aux=False, times=None, differentiable=True, **kwargs):
        """Estimate log-probability of samples x under the flow at time t=1.

        Uses the instantaneous change of variables formula and Hutchinson's
        trace estimator to compute the log-density.

        Args:
            x: Samples at time t=1, shape (B, P, D).
            times: Optional time grid for integration (default: [0, 1]).
            kwargs: Additional arguments passed to `sample_logprob`, e.g.,
                `div_method`, `div_samples`, `method`, `options`.  
        Returns:
            Estimated log-probabilities, shape (B,).
        """
        #convergence_report = self.convergence_test_sample_logprob(x[:100], logprob=None, times=times, reverse=True)
        if return_aux:
            z, trace_int, int_div2, int_v2 = self.integrate_augmented(
                x, times=times, reverse=True, differentiable=differentiable, return_traj=False, **kwargs)
        else:
            z, trace_int, _, _ = self.sample_logprob(x, logprob=None, times=times, differentiable=differentiable, reverse=True,return_traj=False, **kwargs)
        base_logprob = base_dist.log_prob(z)
        logprob = base_logprob - trace_int
        if return_aux:
            return logprob, int_div2, int_v2
        else:
            return logprob
        
