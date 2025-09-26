import torch
from tools import *
from get_args import *
from model import NeRF
import ast
from tqdm import trange
import logging
from torch.utils.tensorboard import SummaryWriter
from data_process.blender import load_blender_data
from data_process.llff import load_llff_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

def train():
    # 初始化一些变量
    parser = get_args()
    args = parser.parse_args()
    base_dir = args.base_dir
    data_type = args.data_type
    train_meta_file = args.train_meta_file
    test_meta_file = args.test_meta_file
    lr = args.lr
    Nc_samples = args.Nc_samples
    Nf_samples = args.Nf_samples
    chunk = args.chunk
    lr_decay = args.lr_decay
    N_rand = args.N_rand
    save_dir = args.checkpoints_save_dir
    save_step = args.checkpoints_save_step
    betas = ast.literal_eval(args.betas)
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - [%(levelname)s] - %(message)s')
    if args.log:
        log_step = args.log_step
        writer = SummaryWriter(log_dir=args.log_dir)

    # 加载数据
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
    H = int(H)
    W = int(W)
    hwf = [H, W, focal]
    
    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    # 将数据进一步处理
    rays = np.stack([get_rays(H, W, K, p) for p in poses[:,:3,:4]], 0)
    rays_rgb = np.concatenate([rays, images[:,None]], 1)
    rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
    rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
    rays_rgb = rays_rgb.astype(np.float32)
    np.random.shuffle(rays_rgb)
    i_batch = 0

    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    rays_rgb = torch.Tensor(rays_rgb).to(device)

    # 创建模型
    params_to_train = []
    coarse_nerf = NeRF(D=args.coarse_net_depth, W=args.coarse_net_width, 
                       in_L=args.multrics_x, in_v_L=args.multrics_v, skips=args.coarse_net_skips).to(device)
    params_to_train.append(coarse_nerf.parameters())
    if args.coarse_net_use_checkpoint:
        logging.info(f'Load coarse net checkpoint ({args.coarse_net_checkpoint}).')
        coarse_nerf_params = torch.load(args.coarse_net_checkpoint, map_location=device)
        coarse_nerf.load_state_dict(coarse_nerf_params)
    
    fine_nerf = NeRF(D=args.fine_net_depth, W=args.fine_net_width, 
                       in_L=args.multrics_x, in_v_L=args.multrics_v, skips=args.fine_net_skips).to(device)
    params_to_train.append(fine_nerf.parameters())
    if args.fine_net_use_checkpoint:
        logging.info(f'Load coarse net checkpoint ({args.fine_net_checkpoint}).')
        fine_nerf_params = torch.load(args.fine_net_checkpoint, map_location=device)
        fine_nerf.load_state_dict(fine_nerf_params)

    optimizer = torch.optim.AdamW(params=params_to_train, lr=lr, betas=betas)

    with trange(args.begin_iter, args.N_iter) as progress_bar:
        for i in progress_bar:
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
            
            rays_o, rays_d = batch_rays
            viewdirs = rays_d
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1,3]).float()

            rays_q, t_vals = uniform_sample_rays(rays_o=rays_o, rays_d=rays_d, N_samples=Nc_samples, near=near, far=far) 

            