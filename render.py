import os
import torch
import numpy as np
from tools import get_rays, uniform_sample_rays, integrate, importance_sample_rays, ndc_rays
import imageio
from tqdm import tqdm

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def render(render_poses, hwf, K, near, far, coarse_nerf, fine_nerf, 
           chunk, save_dir, render_factor, Nc_samples, Nf_samples, ndc):
    # 创建保存路径
    os.makedirs(os.path.join(save_dir), exist_ok=True)
    
    H, W, focal = hwf

    if render_factor != 0:
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor
    
    rgbs = []
    # 不进行梯度计算
    with torch.no_grad():
        for i, c2w in enumerate(tqdm(render_poses, desc="render video")):
            rays_o, rays_d = get_rays(H, W, K, c2w[:3,:4])
            view_dirs = rays_d
            view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)
            view_dirs = torch.reshape(view_dirs, [-1,3]).float()

            if ndc:
                rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

            rays_o = torch.reshape(rays_o, [-1,3]).float()
            rays_d = torch.reshape(rays_d, [-1,3]).float()
            
            rays_query, t_vals = uniform_sample_rays(rays_o=rays_o, rays_d=rays_d, near=near, far=far, N_samples=Nc_samples)    # [batch_size, N_samples, 3], [batch_size, N_samples]
            
            batch_ray_size = rays_query.shape[0]
            batch_samples_size = rays_query.shape[1]
            
            rays_query_flat = rays_query.reshape(-1, 3)
            N_s = rays_query_flat.shape[0]
            view_dirs_flat = view_dirs[:,None,:].expand(rays_query.shape).reshape(-1, 3)
            
            # coarse net
            all_rgb, all_sigma = [], []
            for it in range(0, N_s, chunk):
                begin = it
                end = it + chunk
                end = end if end < N_s else N_s
                rgb, sigma = coarse_nerf(rays_query_flat[begin:end], view_dirs_flat[begin:end])
                all_rgb.append(rgb)
                all_sigma.append(sigma)
            c_rgb = torch.cat(all_rgb, 0).reshape(batch_ray_size, batch_samples_size, 3)
            c_sigma = torch.cat(all_sigma, 0).reshape(batch_ray_size, batch_samples_size)
            _, weights = integrate(c_rgb, c_sigma, rays_d, t_vals, 0.)
            
            # fine net
            rays_query, imp_t_vals = importance_sample_rays(rays_o=rays_o, rays_d=rays_d, t_vals=t_vals, weights=weights, N_imp_samples=Nf_samples)  # [batch_size, N_imp_samples+N_samples, 3]
            
            batch_ray_size = rays_query.shape[0]
            batch_samples_size = rays_query.shape[1]
            
            rays_query_flat = rays_query.reshape(-1, 3)
            N_s = rays_query_flat.shape[0]
            view_dirs_flat = view_dirs[:,None,:].expand(rays_query.shape).reshape(-1, 3)
            
            all_rgb, all_sigma = [], []
            for it in range(0, N_s, chunk):
                begin = it
                end = it + chunk
                end = end if end < N_s else N_s
                rgb, sigma = fine_nerf(rays_query_flat[begin:end], view_dirs_flat[begin:end])
                all_rgb.append(rgb)
                all_sigma.append(sigma)
            f_rgb = torch.cat(all_rgb, 0).reshape(batch_ray_size, batch_samples_size, 3)
            f_sigma = torch.cat(all_sigma, 0).reshape(batch_ray_size, batch_samples_size)
            f_rgb_map, _ = integrate(f_rgb, f_sigma, rays_d, imp_t_vals, 0.)
            
            f_rgb_map = f_rgb_map.reshape(H, W, 3).cpu().numpy()
            
            rgbs.append(f_rgb_map)
        rgbs = np.stack(rgbs, 0)
        imageio.mimwrite(os.path.join(save_dir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
        print("saved video")
