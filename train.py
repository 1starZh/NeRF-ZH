import torch
from tools import *
from get_args import *
import ast
import logging
from torch.utils.tensorboard import SummaryWriter
from data_process.blender import load_blender_data
from data_process.llff import load_llff_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

def train():
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
    
    
    
    logging.info('Init coarse net.')