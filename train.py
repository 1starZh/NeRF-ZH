import torch
from tools import *
from get_args import *
from render import *
from loss import *
from model import NeRF
import ast
from tqdm import trange, tqdm
import time
import logging
from torch.utils.tensorboard import SummaryWriter
from data_process.blender import load_blender_data
from data_process.llff import load_llff_data
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

def train():
    parser = get_args()
    args = parser.parse_args()
    
    # 定义log格式
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - [%(levelname)s] - %(message)s')
    
    if args.log:
        log_step = args.log_step
        writer = SummaryWriter(log_dir=args.log_dir)

    
    # 处理数据
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]
            
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
        
    if args.test:
        render_poses = np.array(poses[i_test])
    render_poses = torch.Tensor(render_poses).to(device)
    
    # 创建保存路径
    basedir = args.result_dir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, "w") as file:
            file.write(open(args.config, 'r').read())
            
    # 初始化模型
    logging.info('Init net.')
    vars_to_train = []
    coarse_nerf = NeRF(D=args.netdepth, W=args.netwidth, 
                       in_L=args.multires, in_v_L=args.multires_views, skips=args.skips).to(device)
    vars_to_train.append(coarse_nerf)
    if args.coarse_net_use_checkpoint:
        logging.info(f'Load coarse net checkpoint ({args.coarse_net_checkpoint}).')
        coarse_nerf_params = torch.load(args.coarse_net_checkpoint, map_location=device)
        coarse_nerf.load_state_dict(coarse_nerf_params)
        
    fine_nerf = NeRF(D=args.netdepth, W=args.netwidth, 
                       in_L=args.multires, in_v_L=args.multires_views, skips=args.skips).to(device)
    vars_to_train.append(fine_nerf)
    if args.fine_net_use_checkpoint:
        logging.info(f'Load coarse net checkpoint ({args.fine_net_checkpoint}).')
        fine_nerf_params = torch.load(args.fine_net_checkpoint, map_location=device)
        fine_nerf.load_state_dict(fine_nerf_params)
    
    # 定义优化器AdamW
    optimizer = torch.optim.AdamW(params=vars_to_train, lr=args.lrate, betas=args.betas)
    
    loss_history = []
    psnr_history = []
    
    # 数据处理，获取光线
    N_rand = args.N_rand
    using_batching = args.using_batching
    
    if using_batching:
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0
        
    if using_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if using_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    for i in trange(args.begin_iter, args.N_iter):
        time0 = time.time()
        if using_batching:
        # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
        else:
            pass
        rays_o, rays_d = batch_rays
        view_dirs = rays_d
        view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)
        rays_query, t_vals = uniform_sample_rays(rays_o=rays_o, rays_d=rays_d, near=near, far=far, N_samples=args.Nc_samples)    # [batch_size, N_samples, 3], [batch_size, N_samples]
        
        batch_ray_size = rays_query.shape[0]
        batch_samples_size = rays_query.shape[1]
        
        rays_query_flat = rays_query.reshape(-1, 3)
        N_s = rays_query_flat.shape[0]
        view_dirs_flat = view_dirs[:,None,:].expand(rays_query.shape).reshape(-1, 3)
        # coarse net
        all_rgb, all_sigma = [], []
        for it in range(0, N_s, args.chunk):
            begin = it
            end = it + arg.chunk
            end = end if end < N_s else N_s
            rgb, sigma = coarse_nerf(rays_query_flat[begin:end], view_dirs_flat[begin:end])
            all_rgb.append(rgb)
            all_sigma.append(sigma)
        c_rgb = torch.cat(all_rgb, 0).reshape(batch_ray_size, batch_samples_size)
        c_sigma = torch.cat(all_sigma, 0).reshape(batch_ray_size, batch_samples_size)
        c_rgb_map, weights = integrate(c_rgb, c_sigma, rays_d, t_vals)
        c_loss = get_mse_loss(c_rgb_map, target_s)  
        
        # fine net
        rays_query, imp_t_vals = importance_sample_rays(rays_o=rays_o, rays_d=rays_d, t_vals=t_vals, weights=weights, N_imp_samples=args.Nf_samples)  # [batch_size, N_imp_samples+N_samples, 3]

        batch_ray_size = rays_query.shape[0]
        batch_samples_size = rays_query.shape[1]
        
        rays_query_flat = rays_query.reshape(-1, 3)
        N_s = rays_query_flat.shape[0]
        view_dirs_flat = view_dirs[:,None,:].expand(rays_query.shape).reshape(-1, 3)
        
        all_rgb, all_sigma = [], []
        for it in range(0, N_s, args.chunk):
            begin = it
            end = it + arg.chunk
            end = end if end < N_s else N_s
            rgb, sigma = fine_nerf(rays_query_flat[begin:end], view_dirs_flat[begin:end])
            all_rgb.append(rgb)
            all_sigma.append(sigma)
        f_rgb = torch.cat(all_rgb, 0).reshape(batch_ray_size, batch_samples_size)
        f_sigma = torch.cat(all_sigma, 0).reshape(batch_ray_size, batch_samples_size)
        f_rgb_map, weights = integrate(f_rgb, f_sigma, rays_d, imp_t_vals)
        f_loss = get_mse_loss(f_rgb_map, target_s)  
        f_psnr = get_psnr(f_loss)
        
        # loss 计算与反向传播
        loss = c_loss + f_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(f_loss.item())
        psnr_history.append(f_psnr.item())
        train_loss = sum(loss_history[-1000:]) / 1000
        train_psnr = sum(psnr_history[-1000:]) / 1000
        
        if (i + 1) % log_step:
            writer.add_scalar("Loss/train", train_loss, i + 1)
            writer.add_scalar("PSRN/train", train_psnr, i + 1)
            
        if (i + 1) % args.save_step == 0:
            save_model_parameters(save_base_dir=args.save_dir, coarse_nerf=coarse_nerf, fine_nerf=fine_nerf, iteration=i+1)

        decay_rate = 0.1
        decay_steps = args.lr_decay * 1000
        new_lrate = args.lr * (decay_rate ** (i / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
            
        dt = time.time() - time0
        
        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {f_psnr.item()}")
        
    writer.close()