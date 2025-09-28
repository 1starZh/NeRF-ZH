import torch
import numpy as np
import os
import logging

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

def importance_sample_rays(rays_o, rays_d, t_vals, weights, N_imp_samples):
    """importance sample rays

    Args:
        rays_o (torch.Tensor): rays origin, [N_rays, 3]
        rays_d (torch.Tensor): rays directory, [N_rays, 3]
        t_vals (torch.Tensor): [N_rays, N_samples]
        weights (torch.Tensor): [N_rays, N_samples]
        N_imp_samples (int): number of importance samples

    Returns:
        torch.Tensor: rays query, [N_rays, N_imp_samples+N_samples, 3]
        torch.Tensor: samples, [N_rays, N_imp_samples+N_samples]
    """
    device = rays_o.device
    weights = weights[...,1:-1] + 1e-5                                              # [N_rays, N_samples-2]
    bins = 0.5 * (t_vals[...,:-1] + t_vals[...,1:])                                 # [N_rays, N_samples-1]
    # integrate pdf to cdf
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)                        # [N_rays, N_samples-2]
    cdf = torch.cumsum(pdf, dim=-1)                                                 # [N_rays, N_samples-2]
    cdf = torch.cat([torch.zeros_like(cdf[...,:1], device=device), cdf], dim=-1)    # [N_rays, N_samples-1]
    # invert transform sampling
    u = torch.rand(list(cdf.shape[:-1]) + [N_imp_samples], device=device)           # uniform sample cdf: [N_rays, N_imp_samples]
    inds = torch.searchsorted(cdf, u, right=True)                                   # bin indices: [N_rays, N_imp_samples]
    below = torch.max(torch.zeros_like(inds-1), inds-1)                             # [N_rays, N_imp_samples]
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)            # [N_rays, N_imp_samples]
    inds_g = torch.stack([below, above], -1)                                        # [N_rays, N_imp_samples, 2]
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]               # matched_shape = [N_rays, N_imp_samples, N_samples-1]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)         # [N_rays, N_imp_samples, 2] <- [N_rays, N_imp_samples, N_samples-1]
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)       # [N_rays, N_imp_samples, 2] <- [N_rays, N_imp_samples, N_samples-1]
    denom = (cdf_g[...,1] - cdf_g[...,0])                                           # [N_rays, N_imp_samples]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)                # [N_rays, N_imp_samples]
    t = (u - cdf_g[...,0]) / denom                                                  # [N_rays, N_imp_samples]
    imp_samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])                 # [N_rays, N_imp_samples]
    imp_samples.detach_()                                                           # [N_rays, N_imp_samples]
    # hierarchical sampling
    samples, _ = torch.sort(torch.cat([imp_samples, t_vals], dim=-1), dim=-1)       # [N_rays, N_imp_samples+N_samples]
    rays_q = rays_o[:,None,:] + samples[...,None] * rays_d[:,None,:]                # [N_rays, N_imp_samples+N_samples, 3] = [N_rays, 1, 3] + [N_rays, N_imp_samples+N_samples, 1] * [N_rays, 1, 3]
    return rays_q, samples

def get_psnr(mse):
    """get psnr from rgb mse

    Args:
        mse (torch.Tensor): rgb mean squared error
    Returns:
        torch.Tensor: psnr
    """
    return -10. * torch.log(mse) / torch.log(torch.Tensor([10.]).to(mse.device))

def save_model_parameters(save_base_dir, coarse_nerf, fine_nerf, iteration):
    save_dir = os.path.join(save_base_dir, f'iter_{iteration}')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(coarse_nerf.state_dict(), os.path.join(save_dir, 'coarse_nerf.pt'))
    torch.save(fine_nerf.state_dict(), os.path.join(save_dir, 'fine_nerf.pt'))
    logging.info(f"Saved model parameters at iteration {iteration}.")