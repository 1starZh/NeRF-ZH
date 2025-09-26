import configargparse

def str2bool(x):
    return x.lower() in ('true')

def get_args():
    parser = configargparse.ArgumentParser()
    
    parser.add_argument('--config', is_config_file=True, required=True, 
                        help='config file path')
    parser.add_argument("--base_dir", type=str, required=True, 
                        help='base directory of model')
    parser.add_argument("--data_type", type=str, required=True,
                        help="data type(for example 'llff' )")
    parser.add_argument("--train_meta_file", type=str, required=True, 
                        help='train meta description file of data')
    
    parser.add_argument("--coarse_net_depth", type=int, default=8, 
                        help='layers in coarse network')
    parser.add_argument("--coarse_net_width", type=int, default=256,
                        help='channels per layer in coarse network')
    parser.add_argument("--coarse_net_skips", nargs='+', type=int, default=[4], 
                        help='layers concat position encoder results in coarse network')
    parser.add_argument("--coarse_net_use_checkpoint", type=str2bool, default=False, 
                        help='coarse network checkpoint file')
    parser.add_argument("--coarse_net_checkpoint", type=str, required=False,  
                        help='coarse network checkpoint file')
    
    parser.add_argument("--fine_net_depth", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--fine_net_width", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--fine_net_skips", nargs='+', type=int, default=[4], 
                        help='layers concat position encoder results in coarse network')
    parser.add_argument("--fine_net_use_checkpoint", type=str2bool, default=False, 
                        help='fine network checkpoint file')
    parser.add_argument("--fine_net_checkpoint", type=str, required=False,  
                        help='fine network checkpoint file')
    
    parser.add_argument("--begin_iter", type=int, default=0, 
                        help='begin of train iterations')
    parser.add_argument("--image_skip", type=int, default=1, 
                        help='skip for image loader')
    parser.add_argument("--N_iter", type=int, default=200000, 
                        help='number of train iterations')
    parser.add_argument("--use_batch", type=str2bool, default=False, 
                        help='use batchified rays for train')
    parser.add_argument("--N_rand", type=int, default=1*1024, 
                        help='number of random rays per gradient step')
    parser.add_argument("--lr", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--betas", type=str, default='(0.9, 0.999)', 
                        help='betas for optimizer')
    parser.add_argument("--lr_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--Nc_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--Nf_samples", type=int, default=128,
                        help='number of additional fine samples per ray')
    parser.add_argument("--multrics_x", type=int, default=10, 
                        help='number of cos&sin function in position encoder for coordinate')
    parser.add_argument("--multrics_v", type=int, default=4, 
                        help='number of cos&sin function in position encoder for directory')
    parser.add_argument("--crop_iters", type=int, default=0, 
                        help='number of iterations to perform center cropping if no batch')
    parser.add_argument("--crop_frac", type=float, default=0.5,
                        help='fraction of the image to use for center cropping if no batch')
    parser.add_argument("--res_half", type=str2bool, default=False, 
                        help='half resolution of images')
    
    parser.add_argument("--checkpoints_save_step", type=int, default=1000, 
                        help='checkpoints save step')
    parser.add_argument("--checkpoints_save_dir", type=str, default='./checkpoints', 
                        help='checkpoints save directory')
    
    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--test_meta_file", type=str, default=False, required=True, 
                        help='test meta description file of blender data')
    parser.add_argument("--test_step", type=int, default=500, 
                        help='frequency of test')
    parser.add_argument("--test_rand_n", type=int, default=1, 
                        help='randomly choose n rays for test')

    parser.add_argument("--log", type=str2bool, default=True, 
                        help='log or not')
    parser.add_argument("--log_dir", type=str, default='./log', 
                        help='loss and metrics log directory')
    parser.add_argument("--log_step", type=int, default=100, 
                        help='frequency of tensorboard logging')
    
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')
    
    return parser