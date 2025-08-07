import torch
import math
import numpy as np
from mpmath import jtheta # wrapped gaussian computations
import mpmath as mp

mp.dps = 25

class BreakAllLoops(Exception):
    pass

def even_spacing(nparticles, boxlength, dim):
    '''
    nparticles: number of particles
    boxlength: boxlength
    '''
    positions = torch.zeros((nparticles, dim))
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
                        positions[idx] = torch.tensor([i, j, k]) * spacing
        elif dim == 2:
            for i in range(num_per_dim):
                for j in range(num_per_dim):
                    idx = i * num_per_dim + j
                    if idx >= nparticles:
                        raise BreakAllLoops()
                    positions[idx] = torch.tensor([i, j]) * spacing

    except BreakAllLoops:
        pass
    return positions

class BaseDistribution(torch.nn.Module):
    def __init__(self, nparticles=4, dim=2, batch_size=10, device='cuda'):
        super(BaseDistribution, self).__init__()
        self.batch_size = batch_size
        self.nparticles = nparticles
        self.dim = dim
        self.param = torch.nn.Parameter(torch.randn(batch_size,nparticles,dim).to(device))

    @property
    def state(self):
        return self.param

    def potential(self, x):
        pass

    def forward(self, x=None):
        return self.log_prob(x)

    def log_prob(self, x=None):
        if x is None:
            x = self.param
        return -self.potential(x)

    def grad_log_prob(self, x=None):
        if x is None:
            x = self.param
        x.requires_grad_(True)
        with torch.enable_grad():
            return  -torch.autograd.grad(-self.potential(x), x,
                                         torch.ones(x.shape[0]).to(x.device), create_graph=True)[0]

    def neg_force_clipped(self, x=None, max_val=80):
        if x is None:
            x = self.param
        return torch.clip(self.grad_log_prob(x),-max_val,max_val)

    def reset_parameters(self):
        self.param.data = torch.randn(self.batch_size,self.nparticles,self.dim).to(self.param.device)

class LJ(BaseDistribution):
    def __init__(self, nparticles=10, dim=3, batch_size=10,
                  device="cuda", boxlength=None, kT=1,
                  epsilon=1., sigma=1., cutoff=None, shift=True,
                  periodic=False, spring_constant=0.5):
        super(LJ, self).__init__(nparticles=nparticles, dim=dim, batch_size=batch_size, device=device)
        self.kT = kT
        self.epsilon=epsilon
        self.sigma=sigma
        self.cutoff=cutoff
        self.shift=shift
        self.boxlength=boxlength
        self.periodic=periodic
        self.spring_constant=spring_constant


    def potential(self,particle_pos, min_dist=None, turn_off_harmonic=False):
        """
        Calculates Lennard_Jones potential between particles
        Arguments:
        particle_pos: A tensor of shape (n_particles, n_dimensions)
        representing the particle positions
        boxlength: A tensor of shape (1) representing the box length
        epsilon: A float representing epsilon parameter in LJ
        Returns:
        total_potential: A tensor of shape (n_particles, n_dimensions)
        representing the total potential of the system
        """
        pair_vec = self.pair_vec(particle_pos)
        distances = torch.linalg.norm(pair_vec.float(), axis=-1)
        rem_dims = distances.shape[:-2]
        n = distances.shape[-1]
        distances = distances.flatten(start_dim=-2)[...,1:].view(*rem_dims,n-1, n+1)[...,:-1].reshape(*rem_dims,n, n-1)
        scaled_distances = distances
        if min_dist is not None:
            scaled_distances = torch.clamp(scaled_distances,min=min_dist)
        distances_inverse = 1/scaled_distances
        if self.cutoff is not None:
            distances_inverse = distances_inverse-(distances >self.cutoff)*distances_inverse
            pow_6 = torch.pow(self.sigma*distances_inverse, 6)
            if self.shift:
                pow_6_shift = (self.sigma/self.cutoff)**6
                pair_potential = self.epsilon * 4 * (torch.pow(pow_6, 2)
                                        - pow_6 - pow_6_shift**2+pow_6_shift)
            else:
                pair_potential = self.epsilon * 4 * (torch.pow(pow_6, 2)
                                        - pow_6)
        else:
            pow_6 = torch.pow(self.sigma*distances_inverse, 6)
            pair_potential = self.epsilon * 4 * (torch.pow(pow_6, 2)
                                        - pow_6)
        pair_potential = pair_potential *distances_inverse*distances
        total_potential = torch.sum(pair_potential,axis=(-1,-2))/2
        if not self.periodic and not turn_off_harmonic:
            total_potential += self.harmonic_potential(particle_pos)
        return total_potential


    def log_likelihood(self,particle_pos, min_dist=None):
        """
        heat version of logprob
        """
        return - self.potential(particle_pos, min_dist=min_dist) / self.kT


    def harmonic_force(self,particle_pos):
        com = torch.mean(particle_pos,axis=-2,keepdim=True)
        rel_pos = particle_pos - com
        harm_f = -self.spring_constant*rel_pos
        return harm_f

    def harmonic_potential(self,particle_pos):
        com = torch.mean(particle_pos,axis=-2,keepdim=True)
        rel_pos = particle_pos - com
        harm_pot = 0.5*self.spring_constant*(rel_pos)**2
        return harm_pot.sum(axis=(-1,-2))

    def grad_log_prob(self, x=None):
        if x is None:
            x = self.param
        return self.force(x)

    def force(self,particle_pos,min_dist=None):
        """
        Calculates Lennard_Jones force between particles
        Arguments:
            particle_pos: A tensor of shape (n_particles, n_dimensions)
        representing the particle positions
        box_length: A tensor of shape (1) representing the box length
        epsilon: A float representing epsilon parameter in LJ

        Returns:
            total_force_on_particle: A tensor of shape (n_particles, n_dimensions)
        representing the total force on a particle
         """
        eps = self.epsilon
        sig = self.sigma
        pair_vec = self.pair_vec(particle_pos)
        distances = torch.linalg.norm(pair_vec.float(), axis=-1)
        scaled_distances = distances + (distances == 0)
        if min_dist is not None:
            scaled_distances = torch.clamp(scaled_distances,min=min_dist)
        distances_inverse = 1/scaled_distances
        if self.cutoff is not None:
            distances_inverse = distances_inverse-(distances >self.cutoff)*distances_inverse
            pow_6 = torch.pow(sig*distances_inverse, 6)
            force_mag = eps * 24 * (2 * torch.pow(pow_6, 2)
                                    - pow_6)*sig*distances_inverse
        else:
            pow_6 = torch.pow(sig/scaled_distances, 6)
            force_mag = eps * 24 * (2 * torch.pow(pow_6, 2)
                                    - pow_6)*sig*distances_inverse
        force_mag = force_mag * distances_inverse
        force = -force_mag.unsqueeze(-1) * pair_vec
        total_force = torch.sum(force, dim=1)
        if not self.periodic:
            total_force += self.harmonic_force(particle_pos)
        return total_force

    # unfortunately this one breaks when particles start to be too far away.
    def pair_vec(self,particle_pos):
        pair_vec = (particle_pos.unsqueeze(-2) - particle_pos.unsqueeze(-3))
        if self.periodic:
            to_subtract = torch.round(pair_vec / self.boxlength) * self.boxlength
            pair_vec -= to_subtract
        else:
            pair_vec = (particle_pos.unsqueeze(-2) - particle_pos.unsqueeze(-3))
        return pair_vec

    def g_r(self,particle_pos, bins=100):
        dim = particle_pos.shape[-1]
        pair_vec = self.pair_vec(particle_pos)
        nsamples = len(pair_vec)
        nparticles = pair_vec.shape[1]
        distances = torch.linalg.norm(pair_vec.float(), axis=-1)
        # remove diagonal zeros
        rem_dims = distances.shape[:-2]
        distances = distances.flatten(start_dim=-2)[...,1:].view(
        *rem_dims,nparticles-1, nparticles+1)[...,:-1].reshape(*rem_dims,nparticles, nparticles-1)
        if self.periodic:
            counts,bins = np.histogram(distances.detach().cpu().numpy(),bins=bins)
            if dim == 2:
                bulk_density = nparticles/(self.boxlength**2)
                areas = math.pi*(bins[1:]**2-bins[:-1]**2)
            elif dim == 3:
                bulk_density = nparticles/(self.boxlength**3)
                areas = 4*math.pi*bins[1:]**2
            g_r = counts/(nparticles-1)**2/nsamples/areas/bulk_density
        else:
            com = torch.mean(particle_pos,axis=1)
            dist_from_com = torch.linalg.norm(particle_pos-com.unsqueeze(1),axis=-1)
            com_atom = torch.min(dist_from_com,axis=1)[1]
            distances = distances[torch.arange(distances.shape[0]),com_atom].flatten()
            counts,bins = np.histogram(distances.detach().cpu().numpy(),bins=bins)
            bulk_density = nparticles/(4/3*math.pi*(self.boxlength/2)**3)
            areas = 4*math.pi*((bins[:-1]+bins[1:])/2)**2*(bins[1:]-bins[:-1])
            g_r = counts/nsamples/areas/bulk_density
        return bins, g_r

    def sample_wrapped_gaussian_1(self, std, size=1, device=None, logprobs = True):
        """
        Sample from an N-D Gaussian and wrap the result into a box of size `wrap`.
        Args:
            mean: array-like, shape (dim,)
            cov: array-like, shape (dim, dim)
            wrap: float, the box size to wrap into (default 2*pi)
            size: int, number of samples
            device: torch.device or None
        Returns:
            samples: torch.Tensor, shape (size, particles, dim)
        """

        # this is a wrapped gaussian pdf
        def f(x, mu):
            q = np.exp(- (std / self.boxlength * 2 * np.pi) ** 2 / 2)
            z = np.pi * (x-mu) / self.boxlength
            return float(jtheta(3, z, q) / (self.boxlength))


        mean = even_spacing(self.nparticles, self.boxlength, self.dim).flatten()
        # print(mean)
        cov = np.eye(mean.shape[0]) * std ** 2
        mean = np.asarray(mean)
        cov = np.asarray(cov)
        # dim = mean.shape[0]
        samples = np.random.multivariate_normal(mean, cov, size=size)
        # print(samples - )
        samples_wrapped = np.mod(samples, self.boxlength)
        samples_wrapped = torch.tensor(samples_wrapped, dtype=torch.float32)
        # print(samples_wrapped.shape, torch.tensor(mean).unsqueeze(0).expand(size, -1).shape)
        # compute sample log prob

        if logprobs:
            logprobs_ = torch.log(torch.tensor([f(s.item(), m.item()) for s, m in zip(samples_wrapped.flatten(), torch.tensor(mean).unsqueeze(0).expand(size, -1).flatten())]))
            logprobs_ = logprobs_.reshape(size, self.nparticles, self.dim).sum(dim=(-1,-2))
        else:
            logprobs_ = torch.zeros(size, device=samples_wrapped.device)

        if device is not None:
            samples_wrapped = samples_wrapped.to(device)
            logprobs_ = logprobs_.to(device)
        return  samples_wrapped.reshape(-1, self.nparticles, self.dim), logprobs_

    def sample_uniform(self, size, device=None):
        """
        Uniformly sample particle configurations inside the box.

        Args:
            size (int): number of configurations
            device: optional torch device

        Returns:
            samples (Tensor): shape (size, nparticles, dim)
            logp (Tensor): shape (size,), uniform log probability
        """
        L = self.boxlength
        P = self.nparticles
        D = self.dim

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        samples = torch.rand(size, P, D, device=device) * L - L / 2

        # Logprob of uniform over box of volume L^PD
        logp = -P * D * torch.log(torch.tensor(L, device=device))
        logp = logp.expand(size)

        return samples, logp
    
    def sample_wrapped_gaussian(self, std, size,  device=None, offsets=None):
        """
        size   : number of configurations
        std    : sigma of the *unwrapped* Gaussian
        """
        L  = self.boxlength
        P  = self.nparticles
        D  = self.dim
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # mean positions: even spacing on the torus
        mean = even_spacing(P, L, D).to(device)  # (P, D)
        cov  = std ** 2 * torch.eye(P * D, device=device)

        # sample and wrap
        flat = torch.distributions.MultivariateNormal(mean.flatten(), cov).sample((size,))
        samples = flat.view(size, P, D).remainder(L)

        mu = mean.expand(size, -1, -1)           # (size,P,D)

        if offsets is None:
            logp = wrapped_normal_logpdf(samples, mu, std, L).sum((-1, -2))
        else:
            # compute logpdf with nearest neighbors
            # no easier way to do here for now.
            logp = wrapped_normal_logpdf_neighbors(samples, mu, std, L, offsets=offsets) #(B O P, D)
            p = torch.exp(logp).reshape(size, -1, P * D)  # (B, O, P * D)
            p = p.prod(dim=-1)  # (B, O )
            logp = torch.log(p.sum(-1))  # (B, O)
            # .sum((-1, -2, -3))

        return samples.to(device), logp.to(device)

    def force_softcore(
        self,
        particle_pos: torch.Tensor,
        lambda_t: torch.Tensor | float,
        a_sc: float = 0.5,           # “α_LJ” soft-core prefactor
        min_dist: float | None = None,
        turn_off_harmonic: bool = False,
        s: float = 2.0,              # power on λ inside softcore term
        t: float = 1.0,              # decay exponent on (1−λ) outside
        n: float = 6.0,              # softness exponent
    ):
        """
        Soft-core (Beutler-style) Lennard-Jones force with time-dependent coupling λ(t).

        U_sc(r;λ) = 4 ε (1−λ)^t * [ (α_LJ λ^s + (r/σ)^n )^{-12/n} − (same)^{-6/n} ]
        F = −∇_r U_sc

        Returns
        -------
        total_force : (B, N, D) tensor  — negative gradient of U_sc
        """
        # Pair vectors & distances
        if torch.isnan(particle_pos).any():
            print("lambda_t:", lambda_t)
            raise ValueError("particle_pos contains NaN values")

        pair_vec = self.pair_vec(particle_pos)                 # (B,N,N,D)
        r = torch.linalg.norm(pair_vec.float(), dim=-1)        # (B,N,N)
        # mask self-interaction to avoid r = 0
        diag_mask = torch.eye(r.size(-1), device=r.device, dtype=torch.bool)
        r = r.masked_fill(diag_mask, 1.0)

        if min_dist is not None:
            r = torch.clamp(r, min=min_dist)
        sigma = self.sigma
        epsilon = self.epsilon
        B = r.shape[0]

        # prepare λ tensor
        λ = torch.as_tensor(lambda_t, device=r.device, dtype=r.dtype)
        λ = λ.view(B, *([1] * (r.ndim - 1)))  # shape (B,1,1)
        λs = λ ** s
        λ_decay = (1.0 - λ) ** t

        # softcore term A = a_sc * λ^s + (r/σ)^n
        r_scaled = r / sigma
        r_scaled_n = r_scaled ** n
        A = a_sc * λs + r_scaled_n
        A = torch.clamp(A, min=1e-12)  # avoid div-by-zero

        # compute force magnitude from −dU/dr
        prefactor = 4 * epsilon * λ_decay * r ** (n - 1) / sigma ** n
        force_mag = prefactor * (
            12.0 * A ** (-12.0 / n - 1) -
            6.0  * A ** (-6.0 / n - 1)
        )

        # unit vector r̂
        r_hat = pair_vec / (r[..., None] + 1e-12)
        force_pairs = -force_mag[..., None] * r_hat            # −∇U_sc
        # # get r value corresponding to max force_mag
        # max_force_idx = torch.argmax(force_mag)
        # r_at_max_force = r.flatten()[max_force_idx]
        # print("r at max force magnitude:", r_at_max_force.item())
        # exit()
        # zero out self-interaction
        force_pairs = force_pairs.masked_fill(diag_mask.unsqueeze(-1), 0.0)

        # sum over j to get net force on i
        total_force = torch.sum(force_pairs, dim=2)            # (B,N,D)

        # optional harmonic confinement
        if (not self.periodic) and (not turn_off_harmonic):
            total_force += self.harmonic_force(particle_pos)

        return total_force





def wrapped_normal_logpdf(x, mu, sigma, L, K=3):
    # ensure sigma and L are tensors that live with x
    sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
    L     = torch.as_tensor(L,     dtype=x.dtype, device=x.device)

    # broadcast everything to the same shape
    x, mu, sigma, L = torch.broadcast_tensors(x, mu, sigma, L)
    ks = torch.arange(-K, K + 1, dtype=x.dtype, device=x.device)     # (2K+1,)

    shifted = x.unsqueeze(-1) + ks * L.unsqueeze(-1)                 # (...,D,2K+1)

    log_gauss = (
        -0.5 * ((shifted - mu.unsqueeze(-1)) / sigma.unsqueeze(-1))**2
        - torch.log(sigma.unsqueeze(-1))
        - 0.5 * torch.log(torch.tensor(2 * torch.pi, dtype=x.dtype, device=x.device))
    )

    logp = torch.logsumexp(log_gauss, dim=-1) - torch.log(L)         # (...,D)
    return logp

def wrapped_normal_logpdf_neighbors(x, mu, sigma, L, K=3, offsets=None):
    """
    offsets : # (nboffset, D)
    """

    if offsets is None:
        return wrapped_normal_logpdf(x, mu, sigma, L, K)
    else:
    # ensure sigma and L are tensors that live with x
        sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
        L     = torch.as_tensor(L,     dtype=x.dtype, device=x.device)



        newmu = mu.unsqueeze(1).expand(-1, offsets.shape[0], -1, -1)
        newmu = newmu + offsets.reshape(1, offsets.shape[0], 1, -1).to(mu)
        newmu = newmu % L  # wrap the means to the box # (B, O ,P,D)

        newx = x.unsqueeze(1).expand(-1, newmu.shape[1], -1, -1)  # (B, O ,P,D)

        # print(newx.shape, newmu.shape, sigma.shape, L.shape)
        # broadcast everything to the same shape
        newx, newmu, sigma, L = torch.broadcast_tensors(newx, newmu, sigma, L)
        ks = torch.arange(-K, K + 1, dtype=x.dtype, device=x.device)     # (2K+1,)


        newshifted = newx.unsqueeze(-1) + ks * L.unsqueeze(-1)        # (B, O ,P,D, 2K+1)


        newlog_gauss = (
            -0.5 * ((newshifted - newmu.unsqueeze(-1)) / sigma.unsqueeze(-1))**2
            - torch.log(sigma.unsqueeze(-1))
            - 0.5 * torch.log(torch.tensor(2 * torch.pi, dtype=x.dtype, device=x.device))
        )

        newlogp = torch.logsumexp(newlog_gauss, dim=-1) - torch.log(L)         # (B, O ,P , D)
        return newlogp