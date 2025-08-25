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

    def divergence_hutch(self, x,t, div_samples=int(1e2 ), **kwargs):
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
        print(kwargs)
        boxlength = kwargs.pop('boxlength', None)

        if 'method' not in  kwargs:
            kwargs['method'] = 'euler'
        # print(kwargs)

        # little reshaping here
        # inorder to have some callback to the forward method we need to define this as a class

        # we do this so we can pass some callbacks
        class IntegrationFunc:
            def __init__(self, model):
                self.model = model

            def __call__(self, t, xs):
                # print('calling')
                t_ = torch.ones(batch_size).to(xs) * t.item()
                return self.model.forward(xs, t_).detach()

        integration_func = IntegrationFunc(self)

        # watch out, I had to modify the core code for callback to act on the state
        if boxlength is not None:

            # this is an inplace modification of xs
            def mod(xs):
                xs = (xs + 0.5 * boxlength) % boxlength - 0.5 * boxlength

            setattr(integration_func, 'callback_step', lambda t, xs, dt: mod(xs)) # this is an inplace operation on xs, which we carry on throught the next integration step

        return odeint(integration_func, x0, times, **kwargs)


  
    # @torch.no_grad()
    def sample_logprob(self, x, logprob=None, times=None, reverse=False, verbose=False, **kwargs):
        """
        ODE integration returning the trajectory and logprob.

        Args:
            x: initial position (x0 if forward, x1 if reverse)
            logprob: initial log probability (if reverse=False)
            times: time vector (optional)
            reverse: if True, integrate from t=1 to t=0
            verbose: print diagnostic info
            kwargs: passed to odeint and divergence
        """
        batch_size = x.shape[0]
        npart = x.shape[-2]
        dim = x.shape[-1]

        boxlength = kwargs.pop('boxlength', None)

        # time setup
        if times is None:
            times = torch.linspace(0, 1, 2).to(x.device)
        if reverse:
            times = torch.flip(times, dims=[0])

        if 'method' not in kwargs:
            kwargs['method'] = 'euler'
        if 'options' not in kwargs:
            kwargs['options'] = {'step_size': 1 / 100}

        # divergence method selection
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
                raise ValueError(f"Unknown divergence method: {kwargs['div_method']}")
            del kwargs['div_method']
        elif hasattr(self, 'divergence'):
            self._divergence = self.divergence
        else:
            print("No divergence method specified, using hutchison trace estimator by default")
            self._divergence = self.divergence_hutch

        if verbose:
            print("Using divergence method:", self._divergence.__name__)
            print("Integrating", "backward" if reverse else "forward")
            
        if logprob is None:
            if reverse:
                logprob = torch.zeros(batch_size, device=x.device)
            else:
                raise ValueError("logprob must be provided for forward sampling.")

        # pack state
        state0 = torch.cat(
            (
                x,
                repeat(
                    logprob,
                    'b -> b p d',
                    p=npart, d=dim
                )
            ),
            dim=0
        )

        class IntegrationFunc:
            def __init__(self, model):
                self.model = model

            def __call__(self, t, state):
                xs = state[:batch_size]
                t_ = torch.full((batch_size,), t.item(), device=xs.device)
                div = self.model._divergence(xs, t_, **div_kwargs).detach()
                v = self.model.forward(xs, t_).detach()
                # flip sign for reverse integration
                vel = -v if reverse else v
                dlogp = div if reverse else -div

                return torch.cat(
                    (
                        vel,
                        repeat(
                            dlogp,
                            'b -> b p d',
                            p=npart, d=dim
                        )
                    ),
                    dim=0
                ).detach()

        integration_func = IntegrationFunc(self)

        # optional: box wrapping per step
        if boxlength is not None:
            def mod(xs):
                xs[:batch_size]  = (xs[:batch_size] + 0.5 * boxlength) % boxlength - 0.5 * boxlength
            setattr(integration_func, 'callback_step', lambda t, xs, dt: mod(xs))

        integrated_state = odeint(integration_func, state0, times, **kwargs)
        all_xs = integrated_state[:, :batch_size]
        all_logprobs = integrated_state[:, batch_size:, 0, 0]

        return all_xs, all_logprobs

    def log_prob(self, x, times=None, verbose=False, **kwargs):
        """Compute the log-likelihood of ``x`` under the flow.

        The method augments the state with a running log-density term and
        integrates the ODE **backward** from ``t=1`` to ``t=0`` using the
        adjoint method provided by :func:`torchdiffeq.odeint_adjoint` for
        memory–efficient differentiation.

        Parameters
        ----------
        x : torch.Tensor
            Data points of shape ``[batch, n_particles, dim]`` evaluated at
            time ``t=1``.
        times : torch.Tensor, optional
            Optional time grid for the integration.  If ``None`` a simple
            two point grid ``[1, 0]`` is used.
        verbose : bool, optional
            If ``True`` prints the divergence method being used.
        **kwargs
            Additional keyword arguments forwarded to the ODE solver.  This
            includes the divergence computation options such as
            ``div_method`` and ``div_samples``.

        Returns
        -------
        x0 : torch.Tensor
            The latent variables at ``t=0``.
        log_prob : torch.Tensor
            The log-density of the input ``x`` up to a constant determined by
            the base distribution.
        """

        batch_size = x.shape[0]
        npart = x.shape[-2]
        dim = x.shape[-1]

        boxlength = kwargs.pop('boxlength', None)

        # ------------------------------------------------------------------
        # Time grid for backward integration
        # ------------------------------------------------------------------
        if times is None:
            times = torch.linspace(0, 1, 2, device=x.device)
        times = torch.flip(times, dims=[0])

        if 'method' not in kwargs:
            kwargs['method'] = 'dopri5'
        if 'options' not in kwargs:
            kwargs['options'] = {'step_size': 1 / 100}

        # ------------------------------------------------------------------
        # Divergence selection (same logic as in ``sample_logprob``)
        # ------------------------------------------------------------------
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
                raise ValueError(f"Unknown divergence method: {kwargs['div_method']}")
            del kwargs['div_method']
        elif hasattr(self, 'divergence'):
            self._divergence = self.divergence
        else:
            self._divergence = self.divergence_full_jacobian

        if verbose:
            print("Using divergence method:", self._divergence.__name__)

        # ------------------------------------------------------------------
        # Initial augmented state: concatenate positions and log-probability
        # ------------------------------------------------------------------
        logprob0 = torch.zeros(batch_size, device=x.device)
        state0 = torch.cat(
            (
                x,
                repeat(
                    logprob0,
                    'b -> b p d',
                    p=npart,
                    d=dim
                )
            ),
            dim=0,
        )

        class IntegrationFunc(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, t, state):
                xs = state[:batch_size]
                t_ = torch.full((batch_size,), t.item(), device=xs.device)
                div = self.model._divergence(xs, t_, **div_kwargs)
                v = self.model.forward(xs, t_)
                vel = -v
                dlogp = div
                return torch.cat(
                    (
                        vel,
                        repeat(
                            dlogp,
                            'b -> b p d',
                            p=npart,
                            d=dim,
                        ),
                    ),
                    dim=0,
                )

        integration_func = IntegrationFunc(self)

        if boxlength is not None:
            def mod(xs):
                xs[:batch_size] = (xs[:batch_size] + 0.5 * boxlength) % boxlength - 0.5 * boxlength
            setattr(integration_func, 'callback_step', lambda t, xs, dt: mod(xs))

        integrated_state = odeint_adjoint(integration_func, state0, times, **kwargs)
        x0 = integrated_state[-1, :batch_size]
        logp = integrated_state[-1, batch_size:, 0, 0]
        return x0, logp
