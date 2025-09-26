import torch
import numpy as np

def getEmbedder(dim, input):
    out = [input]
    for i in range(0, dim):
        sin = torch.sin((2.**i) * input)
        cos = torch.cos((2.**i) * input)
        out.append(sin)
        out.append(cos)
    return out
    
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

def uniform_sample_rays(rays_o, rays_d, N_samples, near=0, far=1.):
    """uniform sample rays

    Args:
        rays_o (torch.Tensor): rays origin, [N_rays, 3]
        rays_d (torch.Tensor): rays directory, [N_rays, 3]
        near (float): near plane Z value
        far (float): far plane Z value
        N_samples (int): number of position samples

    Returns:
       torch.Tensor: rays query, [N_rays, N_samples, 3]
       torch.Tensor: t values, [N_rays, N_samples]
    """
    device = rays_o.device
    N_rays = rays_o.shape[0]
    eta = torch.rand(size=[N_rays, N_samples])                                  # [N_rays, N_samples]
    bins = torch.linspace(near, far, steps=N_samples+1)                         # [N_samples+1]
    lower_bins = bins[None,:-1].expand(size=[N_rays, N_samples])                # [N_samples] -> [N_rays, N_samples]
    upper_bins = bins[None,1:].expand(size=[N_rays, N_samples])                 # [N_samples] -> [N_rays, N_samples]
    t_vals = (lower_bins * (1 - eta) + upper_bins * eta).to(device)             # [N_rays, N_samples]
    rays_q = rays_o[:,None,:] + rays_d[:,None,:] * t_vals[...,None]             # [N_rays, N_samples, 3] = [N_rays, 1, 3] +  [N_rays, 1, 3] * [N_rays, N_samples, 1]
    return rays_q, t_vals

