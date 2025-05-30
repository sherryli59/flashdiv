from .polynomials import hermite
import torch.nn as nn
import torch
from pytorch_lightning import LightningModule
from einops import rearrange, repeat, reduce, einsum
import torch.nn.functional as F
from torch.func import jvp

# base class
class FlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = lambda x, t: torch.cat([x, t], dim=1)

    def forward(self, x, t):
        raise NotImplementedError("Override this method in subclasses")

    @torch.no_grad()
    def divergence(self, x,t, div_samples=int(1e3)):
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

    @torch.no_grad()
    def sample_logprob(self, xs, n_steps: int=100, **kwargs):
        """
        ODE integration returning the last position and associated logprob
        """
        dt = 1. / n_steps
        xs = xs.detach().clone()
        curr_trace = torch.zeros((xs.shape[0])).to(xs)
        batch_size = xs.shape[0]
        for i in range(n_steps):
            t = torch.ones(batch_size).to(xs) * i * dt
            curr_trace += self.divergence(xs, t, **kwargs) * dt
            vt = self.forward(xs, t)
            xs += dt * vt
        return xs, curr_trace

    # to be tested but I think it works
    @torch.no_grad()
    def sample_logprob_fulljac(self, xs, source_log_prob, n_steps = 100, **kwargs):
        """
        ODE integration returning the last position and associated logprob
        The divergence is computed using the full vmapped jacobian computation
        """

        # need to do some weird unsqueeing
        def fwd_unsq(x,t):
            x,t = x.unsqueeze(0),t.unsqueeze(0)
            return self(x,t).squeeze(0)

        jac_fwd_unsq = jacrev(fwd_unsq, argnums=0) # functionnal call, jacrev viz first arg
        bjac_fwd_unsq = vmap(jac_fwd_unsq, in_dims=(0,0)) # batched full jac returns (b,p,d,p,d)

        # compute trace from full batched jacobian
        def trace(x,t):
            fj = bjac_fwd_unsq(x,t)
            # print(fj.shape)
            return einsum(fj, 'b i j i j -> b')


        # print('in here')
        dt = 1. / n_steps
        xs = xs.detach().clone()
        target_log_prob = source_log_prob(xs) #(batch)
        batch_size = xs.shape[0]
        for i in range(n_steps):
            t = torch.ones((batch_size,1)).to(xs) * i * dt
            # print(t.shape)
            div = trace(xs,t)
            target_log_prob -= div * dt
            vt = self.forward(xs, t)
            xs += dt * vt
        return xs, target_log_prob


# Parallel flows
class ParallelFlow(FlowNet):
    def __init__(self, base_flow, nb_heads):
        super().__init__()
        self.base_flow = base_flow
        self.nb_heads = nb_heads
        self.flows = nn.ModuleList(
            [base_flow.instantiate() for _ in range(nb_heads)]
        )


    def forward(self, x, t):
        """
        Inputs
        x :  (batch_size, nbpart, dim) --> This should scale to nbparticles in nd
        t :  (batch_size, 1)

        Outputs
        v :  (batch_size, nbpart, dim)
        """

        xt = torch.zeros_like(x)
        for i in range(self.nb_heads):
            xt += self.flows[i](x, t)
        return xt

    @torch.no_grad()
    def divergence(self, x, t):
        """
        Inputs
        x :  (batch_size, nbpart, dim) --> weird but we want to try it in 2d first
        t :  (batch_size, 1)

        Outputs
        v :  (batch_size)
        """
        div = torch.zeros(x.shape[0]).to(x)
        for i in range(self.nb_heads):
            xt += self.flows[i].divergence(x, t)

        return xt


##### Base velocityNet

class VelocityBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, xt):
        x = self.fc1(xt)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

class VelocityFlow(FlowNet):
    def __init__(self, dim, hidden_dim, num_layers):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Linear(self.dim+1, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, self.dim)
        self.layers = nn.ModuleList([VelocityBlock(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.time_embedding = lambda x, t: torch.cat([x, t], dim=1)

    def forward(self, x, t):
        xt = self.time_embedding(x, t)
        xt = self.encoder(xt)
        for layer in self.layers:
            xt = layer(xt)
        x = self.decoder(xt)
        return x

    @torch.no_grad()
    def divergence(self, x,t, nb_samples=int(1e3)):
        """
        hutchison trace estimator
        """
        x = repeat(x, 'b d -> (b r) d', r=nb_samples)
        t = repeat(t, 'b d -> (b r) d', r=nb_samples)
        v = torch.randn_like(x0).to(x)

        ouptut, jacvp = jvp(lambda x : flow_nvp(x, t),
        (x0,),
        (v,),
        )
        # print(jacvp.shape)

        tr = reduce(
            reduce(
                rearrange(
                    v * jacvp,
                    '(b r) d -> b r d',
                    r=nb_samples),
                'b r d -> b r',
                'sum'),
            'b r  -> b ',
            'mean')
        return tr

### Our RealNVP like attemps

# A LJ MLP (slightly different inputs, but same idea)
class VelocityFlowLJ(FlowNet):
    def __init__(self, dim, hidden_dim, num_layers):
        super().__init__()
        self.dim = dim
        self.encoder = nn.Linear(self.dim+1, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, self.dim)
        self.layers = nn.ModuleList([VelocityBlock(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.time_embedding = lambda x, t: torch.cat([x, t], dim=1)

    def forward(self, x, t):
        xt = x.clone()
        x = rearrange(x, 'b part dim -> b (part dim)')
        xt = self.time_embedding(x, t.view(-1, 1))
        xt = self.encoder(xt)
        for layer in self.layers:
            xt = layer(xt)
        x = self.decoder(xt)
        return rearrange(x, 'b (part dim) -> b part dim', dim=2)


class LinearNVPFlow(FlowNet):
    """
    Functional form : v_x(x,y,t) = x * s(y,t)  + t(y,t)
    """
    def __init__(self, dim, hidden_dim_s,   ):
        super().__init__(dim)
        self.t1 = FlowMLP(self.dim//2, hidden_dim_s, hidden_layer_number)
        self.t2 = FlowMLP(self.dim//2, hidden_dim_s, hidden_layer_number)
        self.s1 = FlowMLP(self.dim//2, hidden_dim_s, hidden_layer_number)
        self.s2 = FlowMLP(self.dim//2, hidden_dim_s, hidden_layer_number)

    def forward(self, x, t):
        x1 = self.time_embedding(x[:,:self.dim//2], t)
        x2 = self.time_embedding(x[:,self.dim//2:], t)
        xt = x.clone()
        xt[:,:self.dim//2] = x[:,:self.dim//2] * self.s2(x2) + self.t2(x2)
        xt[:,self.dim//2:] = x[:,self.dim//2:] * self.s1(x1) + self.t1(x1)

        return xt

    @torch.no_grad()
    def divergence(self, x,t):
        """
        Divergence of the flow field
        """
        x1 = self.time_embedding(x[:,:self.dim//2], t)
        x2 = self.time_embedding(x[:,self.dim//2:], t)
        div = self.s2(x2) + self.s1(x1)

        return div

class PolynomialNVPFlow(FlowNet):
    """
    Functional form : v_x(x,y,t) = \sum_{k=0}^order x ** k * s_k(y,t)
    """
    def __init__(self,dim, hidden_dim_s, hidden_layer_number, order = 2):
        super().__init__(dim)
        self.ms = nn.ModuleList([
            FlowMLP(self.dim//2, hidden_dim_s, hidden_layer_number) for _ in range(order)
        ])
        self.ts = nn.ModuleList([
            FlowMLP(self.dim//2, hidden_dim_s, hidden_layer_number) for _ in range(order)
        ])


    def forward(self, x, t):
        x1 = self.time_embedding(x[:,:self.dim//2], t)
        x2 = self.time_embedding(x[:,self.dim//2:], t)
        xt = x.clone()

        xt[:,:self.dim//2] = self.ms[0](x2)
        xt[:,self.dim//2:] = self.ts[0](x1)
        for k in range(1,len(self.ms)):
            xt[:,:self.dim//2] += x[:,:self.dim//2] ** k / k * self.ms[k](x2) # divide by k to ease divergence
            xt[:,self.dim//2:] += x[:,self.dim//2:] ** k / k * self.ts[k](x1)

        return xt

    @torch.no_grad()
    def divergence(self, x,t):
        """
        Divergence of the flow field
        """
        x1 = self.time_embedding(x[:,:self.dim//2], t)
        x2 = self.time_embedding(x[:,self.dim//2:], t)

        div = zeros((x.shape[0], 1)).to(x)
        for k in range(1,len(self.ms)):
            div += x[:,:self.dim//2] ** (k-1) * self.ms[k](x2) + x[:,self.dim//2:] ** (k-1)  * self.ts[k](x1)

        return div

class MLPNVPFlow(FlowNet):
    """
    Functional form : v_x(x,y,t) = \sum_{k=0}^order mlp_k(x) * s_k(y,t) --> probably the most expressive but the divergence will be harder to compute
    """
    def __init__(self, dim, hidden_dim_s, hidden_layer_number, order):
        super().__init__(dim)

        self.mlp1 = nn.ModuleList([
            MLP(dim//2, hidden_dim_s, hidden_layer_number) for _ in range(order)
        ])
        self.mlp2 = nn.ModuleList([
            MLP(dim//2, hidden_dim_s, hidden_layer_number) for _ in range(order)
        ])

        self.ms = nn.ModuleList([
            FlowMLP(self.dim//2, hidden_dim_s, hidden_layer_number) for _ in range(order)
        ])
        self.ts = nn.ModuleList([
            FlowMLP(self.dim//2, hidden_dim_s, hidden_layer_number) for _ in range(order)
        ])


    def forward(self, x, t):
        x1 = self.time_embedding(x[:,:self.dim//2], t)
        x2 = self.time_embedding(x[:,self.dim//2:], t)
        xt = x.clone()
        xt = torch.zeros_like(x)

        for k in range(len(self.ms)):
            xt[:,:self.dim//2] += self.mlp1[k](x[:,:self.dim//2]) * self.ms[k](x2)
            xt[:,self.dim//2:] += self.mlp2[k](x[:,self.dim//2:]) * self.ts[k](x1)


        return xt


    @torch.no_grad()
    def divergence(self, x,t):
        raise NotImplementedError("Divergence not yet implemented for MLPNVPFlow")

class PolyMLPNVPFlow(FlowNet):
    """
    Functional form : v_x(x,y,t) = \sum_{k=0}^order (mlp(x)) ** k * s_k(y,t) --> This one wouldn't cost too much to compute the divergence
    """
    def __init__(self, dim, hidden_dim_s, hidden_layer_number, order):
        super().__init__(dim)
        self.order = order
        self.mlp1 = MLP(dim//2, hidden_dim_s, hidden_layer_number)
        self.mlp2 = MLP(dim//2, hidden_dim_s, hidden_layer_number)
        self.ms = nn.ModuleList([
            FlowMLP(self.dim//2, hidden_dim_s, hidden_layer_number) for _ in range(self.order +1)
        ])
        self.ts = nn.ModuleList([
            FlowMLP(self.dim//2, hidden_dim_s, hidden_layer_number) for _ in range(self.order +1)
        ])


    def forward(self, x, t):
        x1 = self.time_embedding(x[:,:self.dim//2], t)
        x2 = self.time_embedding(x[:,self.dim//2:], t)
        xt = torch.zeros_like(x)
        mlp1 = self.mlp1(x[:,:self.dim//2])
        mlp2 = self.mlp2(x[:,self.dim//2:])

        for k in range(self.order + 1):
            xt[:,:self.dim//2] += mlp1 ** k * self.ms[k](x2)
            xt[:,self.dim//2:] += mlp2 ** k * self.ts[k](x1)

        return xt


    @torch.no_grad()
    def divergence(self, x,t):
        raise NotImplementedError("Divergence not yet implemented for PolyMLPNVPFlow")


### helper modules

class FlowMLP(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim+1, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers-1)] + [nn.Linear(hidden_dim, dim)])
        self.activation = nn.ReLU()


    def forward(self, x):
        xt = x.clone()
        for layer in self.layers[:-1]:
            xt = self.activation(layer(xt))
        xt = self.layers[-1](xt)
        return xt

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers-1)] + [nn.Linear(hidden_dim, dim)])
        self.activation = nn.ReLU()


    def forward(self, x):
        xt = x.clone()
        for layer in self.layers[:-1]:
            xt = self.activation(layer(xt))
        xt = self.layers[-1](xt)
        return xt

class HermiteFlow(FlowNet):
    """
    Functional form : v_x(x,y,t) = \sum_{k=0}^order H_k(x) * s_k(y,t)
    """
    def __init__(self,dim, hidden_dim_s, hidden_layer_number, order = 2):
        super().__init__(dim)
        self.order = order
        self.ms = nn.ModuleList([
            FlowMLP(self.dim//2, hidden_dim_s, hidden_layer_number) for _ in range(self.order+1)
        ])
        self.ts = nn.ModuleList([
            FlowMLP(self.dim//2, hidden_dim_s, hidden_layer_number) for _ in range(self.order+1)
        ])


    def forward(self, x, t):
        x1 = self.time_embedding(x[:,:self.dim//2], t)
        x2 = self.time_embedding(x[:,self.dim//2:], t)
        xt = x.clone()

        xt = torch.zeros_like(x)
        hermite_1 = hermite(x[:,0], self.order)
        hermite_2 = hermite(x[:,1], self.order)
        for k in range(self.order+1):
            xt[:,:self.dim//2] += hermite_1[:,k].unsqueeze(1) * self.ms[k](x2)
            xt[:,self.dim//2:] += hermite_2[:,k].unsqueeze(1) * self.ts[k](x1)

        return xt

    @torch.no_grad()
    def divergence(self, x,t):
        """
        Divergence of the flow field
        """
        raise NotImplementedError("Divergence not implemented for Hermite flow")