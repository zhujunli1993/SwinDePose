import argparse
import os

def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))
class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Datasets related
        g_data = parser.add_argument_group('Data')
        g_data.add_argument('--dataset_name', type=str, default='ycb', help='dataset name: ycb | linemod')
        g_data.add_argument('--linemod_cls', type=str, default='ape', help='linemod class label: ape, benchvise, cam, can, cat, driller\
                                            duck,eggbox, glue, holepuncher, iron, lamp, phone')
        g_data.add_argument('--syn', action='store_true', help='Generate YCBV synthetic pseudo images')
        g_data.add_argument('--real', action='store_true', help='Generate YCBV real pseudo images')
        g_data.add_argument('--width', type=int, default=640, help='YCBV image width')
        g_data.add_argument('--height', type=int, default=480, help='YCBV image height')
        g_data.add_argument('--data_root', type=str, help='Dataset Location.')
        g_data.add_argument('--rm_outline', action='store_true', help='Remove points with too small depth value from depth image.')
        g_data.add_argument('--angles_gen', type=str,  help='The C++ codes directory for generating XYZ angles and signed angles file.')
        g_data.add_argument('--train_list', type=str,  help='training list.')
        g_data.add_argument('--test_list', type=str,  help='testing list.')
        g_data.add_argument('--lm_no_fuse', action='store_true', help='Not including fuse/ data when training as data augmentation.')
        g_data.add_argument('--lm_no_render', action='store_true', help='Not including render/ data when training as data augmentation.')
        
        # Experiment related
        g_exp = parser.add_argument_group('Experiment')
        g_exp.add_argument('--exp_dir', type=str, default='/workspace/REPO/pose_estimation', help='code directory')
        g_exp.add_argument('--wandb_proj', type=str, help='wandb project name')
        g_exp.add_argument('--wandb_name', type=str, help='wandb run name')
        g_exp.add_argument('--gpu_id', type=int, default=0, help='gpu id for cuda')
        g_exp.add_argument('--num_threads', default=4, type=int, help='# sthreads for loading data')
        g_exp.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        g_exp.add_argument('--pin_memory', action='store_true', help='pin_memory')
        g_exp.add_argument('--local_rank', type=int, default=0)
        g_exp.add_argument('--nodes', default=1, type=int, metavar='N')
        g_exp.add_argument('--gpus', default=1, type=int,help='number of gpus per node')
        g_exp.add_argument('--nr', default=1, type=int,help='ranking within the nodes')
        g_exp.add_argument('--gpu', type=str, default="0")
        g_exp.add_argument('--deterministic', action='store_true')
        g_exp.add_argument('--keep_batchnorm_fp32', default=True)
        g_exp.add_argument('--opt_level', default="O0", type=str,help='opt level of apex mix presision trainig.')
        g_exp.add_argument('--depth_only', action='store_true', help='if true, replace pseudo to pure depth image')
        g_exp.add_argument('--add_depth', action='store_true', help='if true, add depth to input')
        g_exp.add_argument('--depth_split', action='store_true', help='if true, add depth to input and consider depth channel as a seperate info')
        
        
        # Training related
        g_train = parser.add_argument_group('Training')
        g_train.add_argument('--log_eval_dir', type=str, help='path to save evaluation results' )
        g_train.add_argument('--save_checkpoint', type=str, help='path to save checkpoints')
        g_train.add_argument('--load_checkpoint', type=str, help='path to load checkpoints')
        g_train.add_argument('--log_traininfo_dir', type=str, help='path to save training losses results')
        g_train.add_argument('--n_total_epoch', type=int, default=25, help='epoch numbers')
        g_train.add_argument('--mini_batch_size', type=int, default=8, help='training batch size')
        g_train.add_argument('--val_mini_batch_size', type=int, default=8, help='validation batch size')
        g_train.add_argument('--test_mini_batch_size', type=int, default=1, help='testing batch size')
        g_train.add_argument('--weight_decay', type=float, default=0,help='L2 regularization coeff [default: 0.0]')
        g_train.add_argument('--lr', type=float, default=1e-2, help="Initial learning rate [default: 1e-2]")
        g_train.add_argument('--lr_decay', type=float, default=0.5,help="Learning rate decay gamma [default: 0.5]")
        g_train.add_argument('--decay_step', type=float, default=2e5,help="Learning rate decay step [default: 20]")
        g_train.add_argument('--bn_momentum', type=float, default=0.9,help="Initial batch norm momentum [default: 0.9]")
        g_train.add_argument('--bn_decay', type=float, default=0.5,help="Batch norm momentum decay gamma [default: 0.5]")
        
        # for eval
        parser.add_argument('--val_test_error', action='store_true', help='validate errors of test data')
        parser.add_argument('--eval_net', action='store_true', help="whether is to eval net.")

        
        # Testing related
        g_test = parser.add_argument_group('Testing')
        g_test.add_argument('--resolution', type=int, default=512, help='# of grid in mesh reconstruction')
        g_test.add_argument('--test', action="store_true")
        g_test.add_argument('--test_pose', action="store_true")
        g_test.add_argument('--test_gt', action="store_true")
        g_test.add_argument('--cal_metrics', action="store_true")
        g_test.add_argument('--view_dpt', action="store_true")
        g_test.add_argument('--debug', action='store_true')
        
        # Sampling related
        g_sample = parser.add_argument_group('Sampling')
        g_sample.add_argument('--n_sample_points', type=int, default=480*640//24, help='number of input points, for ycbv dataset')
        g_sample.add_argument('--n_keypoints', type=int, default=8, help='number of keypoints')
        g_sample.add_argument('--n_min_points', type=int, default=400, help='??')
        g_sample.add_argument('--noise_trans', type=float, default=0.05, help='range of the random noise of translation added to the training data')
        
        
        
        # Model related
        g_model = parser.add_argument_group('Model')
        
        # General
        g_model.add_argument('--resnet_ptr_mdl_p', type=str, default='/workspace/pose-estimation-dls1/ffb6d/models/cnn/ResNet_pretrained_mdl', help='ResNet pretrained model path.')
        g_model.add_argument('--full', action='store_true', help='using both angles and signed angles.')
        # RandLA
        g_model.add_argument('--k_n', type=int, default=16, help='KNN')
        g_model.add_argument('--num_layers', type=int, default=4, help='Number of layers')
        # g_model.add_argument('--num_points', type=int, default=480 * 640 // 24, help='Number of input points')
        # g_model.add_argument('--num_classes', type=int, default=22, help='ycb:22 | lm:2')
        g_model.add_argument('--sub_grid_size', type=float, default=0.06, help='preprocess_parameter')
        g_model.add_argument('--batch_size', type=int, default=3, help='batch_size during training')
        g_model.add_argument('--val_batch_size', type=int, default=3, help='batch_size during validation and test')
        g_model.add_argument('--train_steps', type=int, default=500, help='Number of steps per epochs')
        g_model.add_argument('--val_steps', type=int, default=100, help='Number of validation steps per epoch')
        g_model.add_argument('--in_c', type=int, default=12, help='Number of validation steps per epoch')
        g_model.add_argument('--sub_sampling_ratio', nargs='+', default=[4,4,4,4], type=int, help='sampling ratio of random sampling at each layer')
        g_model.add_argument('--d_out', nargs='+', default=[32, 64, 128, 256], type=int, help='feature dimension')
        # g_model.add_argument('--num_sub_points', nargs='+', default=[480 * 640 // 24 // 4, 480 * 640 // 24 // 16, 480 * 640 // 24 // 64, 480 * 640 // 24 // 256],  
        #                      help='num_points // 4, num_points // 16, num_points // 64, num_points // 256')        
        
        # aug
        # group_aug = parser.add_argument_group('aug')
        # group_aug.add_argument('--aug_alstd', type=float, default=0.0, help='augmentation pca lighting alpha std')
        # group_aug.add_argument('--aug_bri', type=float, default=0.0, help='augmentation brightness')
        # group_aug.add_argument('--aug_con', type=float, default=0.0, help='augmentation contrast')
        # group_aug.add_argument('--aug_sat', type=float, default=0.0, help='augmentation saturation')
        # group_aug.add_argument('--aug_hue', type=float, default=0.0, help='augmentation hue')
        # group_aug.add_argument('--aug_blur', type=float, default=0.0, help='augmentation blur')

        # special tasks
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        return opt
