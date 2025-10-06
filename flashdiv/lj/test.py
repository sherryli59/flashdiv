import torch
import numpy as np
import matplotlib.pyplot as plt
from flashdiv.lj.lj import LJ  # replace with actual import if needed

# ---- MALA Sampler ---- #
def run_mala(target, x_init, n_steps, dt, burn_in=1000, save_every=10):
    x = x_init.clone()
    force = target.force(x)
    potential = target.potential(x)
    samples = []

    for i in range(n_steps + burn_in):
        noise = torch.randn_like(x) * np.sqrt(2 * dt * target.kT)
        x_new = x + dt * force + noise
        potential_new = target.potential(x_new)
        force_new = target.force(x_new)

        ratio = potential - potential_new
        ratio -= ((x - x_new - dt * force_new) ** 2).sum(dim=(-1, -2)) / (4 * dt * target.kT)
        ratio += ((x_new - x - dt * force) ** 2).sum(dim=(-1, -2)) / (4 * dt * target.kT)
        ratio = torch.exp(ratio)

        accept = torch.rand_like(ratio) < torch.min(ratio, torch.ones_like(ratio))
        x[accept] = x_new[accept]
        force[accept] = force_new[accept]
        potential[accept] = potential_new[accept]

        if i >= burn_in and (i - burn_in) % save_every == 0:
            samples.append(x.clone())

    return torch.stack(samples, dim=0)

# ---- MH Sampler ---- #
def run_mh(target, x_init, n_steps, dt, burn_in=1000, save_every=10):
    x = x_init.clone()
    samples = []

    for i in range(n_steps + burn_in):
        x_prop = x.clone()
        batch_size, n_particles, dim = x.shape

        # Propose: one particle per batch gets random move
        particle_idx = torch.randint(0, n_particles, (batch_size,))
        noise = dt * torch.randn_like(x[:, 0, :])
        x_prop[torch.arange(batch_size), particle_idx] += noise

        pot = target.potential(x)
        pot_prop = target.potential(x_prop)
        ratio = torch.exp((pot - pot_prop) / target.kT)

        accept = torch.rand_like(ratio) < torch.min(ratio, torch.ones_like(ratio))
        x[accept] = x_prop[accept]

        if i >= burn_in and (i - burn_in) % save_every == 0:
            samples.append(x.clone())

    return torch.stack(samples, dim=0)

# ---- ESS Estimator ---- #
def compute_ess(samples, max_lag=100):
    samples = samples - samples.mean()
    n = samples.shape[0]

    def autocorr(lag):
        if lag >= n:
            return 0.0
        return torch.dot(samples[:-lag], samples[lag:]) / torch.dot(samples, samples)

    rho = torch.tensor([autocorr(lag) for lag in range(1, max_lag)])
    tau = 1.0
    for i in range(0, len(rho) - 1, 2):
        if rho[i] + rho[i + 1] < 0:
            break
        tau += 2 * (rho[i] + rho[i + 1])
    return n / tau

# ---- Run on Your LJ System ---- #
torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
target = LJ(nparticles=10, dim=2, batch_size=128, device=device, kT=1.0, boxlength=5.0, cutoff=2.5, shift=True)

x0 = torch.randn(128, 10, 2).to(device)

samples_mala = run_mala(target, x0.clone(), n_steps=5000, dt=0.01)
samples_mh = run_mh(target, x0.clone(), n_steps=5000, dt=0.1)

# Compute energies
energy_mala = target.potential(samples_mala.view(-1, 10, 2))
energy_mh = target.potential(samples_mh.view(-1, 10, 2))

# Plot
#plt.hist(energy_mala.cpu().numpy(), bins=100, alpha=0.6, label='MALA')
plt.hist(energy_mh.cpu().numpy(), bins=100, alpha=0.6, label='MH')
plt.legend()
plt.xlabel('Energy')
plt.ylabel('Frequency')
plt.title('LJ Energy Distributions: MALA vs MH')
plt.savefig("lj_energy_comparison.png")
plt.show()

# ESS
ess_mala = compute_ess(energy_mala.cpu())
ess_mh = compute_ess(energy_mh.cpu())

print(f"ESS (MALA): {ess_mala:.2f}")
print(f"ESS (MH):   {ess_mh:.2f}")
