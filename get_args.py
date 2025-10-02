import configargparse

def str2bool(x):
    return x.lower() in ('true')

def get_args():
    parser = configargparse.ArgumentParser()
    
    # basic
    parser.add_argument('--config', is_config_file=True, required=True, 
                        help='config file path')
    parser.add_argument("--base_dir", type=str, required=True, 
                        help='base directory of model')
    parser.add_argument("--dataset_type", type=str, required=True,
                        help="data type(for example 'llff' )")
    
    # log
    parser.add_argument("--log", type=str2bool, default=True, 
                        help='log or not')
    parser.add_argument("--log_dir", type=str, default='./log', 
                        help='loss and metrics log directory')
    parser.add_argument("--log_step", type=int, default=100, 
                        help='frequency of tensorboard logging')
    parser.add_argument("--result_dir", type=str, default="./result/",
                        help='where to store ckpts and logs')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    
    # model
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--skips", nargs='+', type=int, default=[4], 
                        help='layers concat position encoder results in network')
    parser.add_argument("--coarse_net_use_checkpoint", type=str2bool, default=False, 
                        help='coarse network checkpoint file')
    parser.add_argument("--coarse_net_checkpoint", type=str, required=False,  
                        help='coarse network checkpoint file')
    parser.add_argument("--fine_net_use_checkpoint", type=str2bool, default=False, 
                        help='fine network checkpoint file')
    parser.add_argument("--fine_net_checkpoint", type=str, required=False,  
                        help='fine network checkpoint file')

    parser.add_argument("--betas", type=str, default='(0.9, 0.999)', 
                        help='betas for optimizer')
    
    # 数据
    parser.add_argument("--datadir", type=str, default='./data/fern', 
                        help='input data directory')
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')
    
    # test
    parser.add_argument("--test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--test_step", type=int, default=500, 
                        help='frequency of test')
    parser.add_argument("--test_rand_n", type=int, default=1, 
                        help='randomly choose n rays for test')
    
    # 训练
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--using_batching", action='store_true', 
                        help='random images for rays')
    parser.add_argument("--begin_iter", type=int, default=0, 
                        help='begin of train iterations')
    parser.add_argument("--image_skip", type=int, default=1, 
                        help='skip for image loader')
    parser.add_argument("--N_iter", type=int, default=200000, 
                        help='number of train iterations')
    parser.add_argument("--Nc_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--Nf_samples", type=int, default=128,
                        help='number of additional fine samples per ray')
    parser.add_argument("--chunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lr_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 
    
    # 恢复训练参数
    parser.add_argument("--resume", action='store_true', 
                        help='是否从之前的检查点恢复训练')
    parser.add_argument("--resume_dir", type=str, default=None, 
                        help='指定恢复训练的检查点目录，默认使用save_dir')
    
    # saving options
    parser.add_argument("--save_step", type=int, default=1000, 
                        help='checkpoints save step')
    parser.add_argument("--save_dir", type=str, default='./checkpoints', 
                        help='checkpoints save directory')
    
    # render
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')
    parser.add_argument("--render_video", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--save_dir_test", type=str, default='./render_result', 
                        help='video save directory')
    parser.add_argument("--render_factor", type=int, default=8, 
                        help='downsample factor for LLFF images')    

    return parser