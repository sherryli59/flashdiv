from einops import rearrange, repeat, reduce
import torch
import numpy as np


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