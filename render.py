import torch

def integrate(rgb, sigma, rays_d, t_vals):
    """integrate rgb, alpha, rays_d and t_vals to rgb_map

    Args:
        rgb (torch.Tensor): [N_rays, N_samples, 3]
        sigma (torch.Tensor): [N_rays, N_samples]
        rays_d (torch.Tensor): [N_rays, 3]
        t_vals (torch.Tensor): [N_rays, N_samples]

    Returns:
        torch.Tensor: rgb_map, [N_rays, 3]
        torch.Tensor: weights, [N_rays, N_samples]
    """
    device = t_vals.device
    dists = t_vals[...,1:] - t_vals[...,:-1]                                                                            # [N_rays, N_samples-1]
    dists = torch.cat([dists, torch.tensor([1e10], device=device).expand(dists[...,:1].shape)], -1)                     # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)                                                              # [N_rays, N_samples] = [N_rays, N_samples] * [N_rays, 1]
    alpha = 1 - torch.exp(-sigma * dists)                                                                               # [N_rays, N_samples]
    T = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1]        # [N_rays, N_samples]
    weights = alpha * T                                                                                                 # [N_rays, N_samples]
    rgb_map = torch.sum(rgb * weights[...,None], dim=1)                                                                 # [N_rays, N_samples, 3] -> [N_rays, 3]
    return rgb_map, weights