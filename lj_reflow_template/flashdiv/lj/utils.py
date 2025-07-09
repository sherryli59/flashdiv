from einops import rearrange, repeat, reduce
import torch
import numpy as np
from scipy.stats import multivariate_normal


def twod_rotation(lj_data, nbrot):
    """
    creates more samples by randomly rotating the centeredparticles in 2D

    Args:
        lj_data: [batch_size, nbparticles, dim]
        nbrot: number of rotations to create
    Returns:
        target_data_new: [batch_size * nbrot, nbparticles, dim]
    """
    nbparticles = lj_data.shape[1]
    dim = lj_data.shape[2]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target_data = lj_data.clone().to(device)
    target_data = target_data[torch.arange(target_data.size(0)).unsqueeze(-1), torch.argsort(target_data[:, :, 0], dim=1)] # sort by x
    target_cm = repeat(reduce(target_data, 'b p d -> b 1 d', 'mean'), 'b 1 d -> b p d', p=nbparticles)
    target_data = target_data - target_cm


    target_data_c = target_data[:,:,0] + 1j * target_data[:,:,1] # go to complex numbers
    target_data_c = repeat(target_data_c, 'b p -> b k p ', k=nbrot)
    rotations = repeat(torch.exp(torch.randn(target_data_c.shape[:-1]).to(device) * 2 * 1j * np.pi), 'b k -> b k p', p=nbparticles)
    target_data_c = target_data_c * rotations
    target_data_new = torch.zeros(target_data.shape[0], nbrot, nbparticles, dim).to(target_data)
    target_data_new[:,:,:,0] = target_data_c.real
    target_data_new[:,:,:,1] = target_data_c.imag
    target_data_new = rearrange(target_data_new, 'b k p d -> (b k) p d ')

    return target_data_new

def make_periodic(x, boxlength):
    """
    Make the input periodic
    """
    x = x.clone()
    x = x % boxlength
    return x

def threed_rotation(lj_data, nbrot):
    """
    Creates more samples by randomly rotating the centered particles in 3D
    Args:
        lj_data: [batch_size, nbparticles, dim] where dim=3
        nbrot: number of rotations to create
    Returns:
        target_data_new: [batch_size * nbrot, nbparticles, dim]
    """
    nbparticles = lj_data.shape[1]
    dim = lj_data.shape[2]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    target_data = lj_data.clone().to(device)

    # Sort by x coordinate
    target_data = target_data[torch.arange(target_data.size(0)).unsqueeze(-1),
                             torch.argsort(target_data[:, :, 0], dim=1)]

    # Center the data by subtracting center of mass
    target_cm = repeat(reduce(target_data, 'b p d -> b 1 d', 'mean'),
                      'b 1 d -> b p d', p=nbparticles)
    target_data = target_data - target_cm

    # Expand data for multiple rotations
    target_data_expanded = repeat(target_data, 'b p d -> b k p d', k=nbrot)

    # Generate random rotation matrices for each batch and rotation
    batch_size = target_data.shape[0]
    rotation_matrices = generate_random_rotation_matrices(batch_size, nbrot, device)

    # Apply rotations: [b, k, p, d] @ [b, k, d, d] -> [b, k, p, d]
    target_data_new = torch.einsum('bkpd,bkde->bkpe', target_data_expanded, rotation_matrices)

    # Reshape to final format
    target_data_new = rearrange(target_data_new, 'b k p d -> (b k) p d')

    return target_data_new

def generate_random_rotation_matrices(batch_size, nbrot, device):
    """
    Generate random 3D rotation matrices using quaternions
    Args:
        batch_size: number of batches
        nbrot: number of rotations per batch
        device: torch device
    Returns:
        rotation_matrices: [batch_size, nbrot, 3, 3]
    """
    # Generate random quaternions (4D unit vectors)
    q = torch.randn(batch_size, nbrot, 4, device=device)
    q = q / torch.norm(q, dim=-1, keepdim=True)  # Normalize to unit quaternions

    # Extract quaternion components
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Convert quaternions to rotation matrices
    rotation_matrices = torch.zeros(batch_size, nbrot, 3, 3, device=device)

    # Fill rotation matrix elements
    rotation_matrices[..., 0, 0] = 1 - 2 * (q2**2 + q3**2)
    rotation_matrices[..., 0, 1] = 2 * (q1 * q2 - q0 * q3)
    rotation_matrices[..., 0, 2] = 2 * (q1 * q3 + q0 * q2)

    rotation_matrices[..., 1, 0] = 2 * (q1 * q2 + q0 * q3)
    rotation_matrices[..., 1, 1] = 1 - 2 * (q1**2 + q3**2)
    rotation_matrices[..., 1, 2] = 2 * (q2 * q3 - q0 * q1)

    rotation_matrices[..., 2, 0] = 2 * (q1 * q3 - q0 * q2)
    rotation_matrices[..., 2, 1] = 2 * (q2 * q3 + q0 * q1)
    rotation_matrices[..., 2, 2] = 1 - 2 * (q1**2 + q2**2)

    return rotation_matrices

def com_logprob(x):

    assert x.dtype == torch.float64, "Input tensor must be of type torch.float64 for numerical stability."
    # whatch out, need to cast as torch.float64 to avoid numerical issues
    nbparticles = x.shape[-2]
    dim = x.shape[-1]


    # define the singular covariance matrix
    bv = repeat(
        torch.tensor([1,0,0], dtype=float), # if we don't set this it fails because numpy conversion sends to float32 negative values
        'd -> (p d)',
        p = nbparticles
        )

    com_matrix = repeat(
        torch.vstack([bv, bv.roll(1,0), bv.roll(2,0)]),
        'p d -> (b p) d',
        b = nbparticles
        )
    com_matrix = (-1 / nbparticles) * com_matrix
    lin_matrix = torch.eye(com_matrix.shape[0]) + com_matrix

    singular_cov = torch.matmul(lin_matrix, lin_matrix.T)
    singular_cov = np.array(singular_cov)



    dist = multivariate_normal(cov=singular_cov , allow_singular=True)

    return dist.logpdf(x.view(-1, nbparticles * dim))