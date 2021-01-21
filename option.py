import torchvision.models as models
import argparse
import sys


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Regularization-Pruning PyTorch')
parser.add_argument('--data', metavar='DIR', # @mst: 'data' -> '--data'
                    help='path to dataset')
parser.add_argument('--dataset',
                    help='dataset name', choices=['mnist', 'cifar10', 'cifar100', 'imagenet', 'imagenet_subset_200', 'tiny_imagenet'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    # choices=model_names, # @mst: We will use more than the imagenet models, so remove this
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
# @mst
import os
from utils import strlist_to_list, strdict_to_dict, check_path, parse_prune_ratio_vgg, merge_args
from model import num_layers, is_single_branch
pjoin = os.path.join

# routine params
parser.add_argument('--project_name', type=str, default="")
parser.add_argument('--debug', action="store_true")
parser.add_argument('--screen_print', action="store_true")
parser.add_argument('--note', type=str, default='', help='experiment note')
parser.add_argument('--print_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=2000)
parser.add_argument('--plot_interval', type=int, default=100000000)
parser.add_argument('--save_interval', type=int, default=2000, help="the interval to save model")
parser.add_argument('--params_json', type=str, default='', help='experiment params')

# base model related
parser.add_argument('--resume_path', type=str, default=None, help="supposed to replace the original 'resume' feature")
parser.add_argument('--directly_ft_weights', type=str, default=None, help="the path to a pretrained model")
parser.add_argument('--base_model_path', type=str, default=None, help="the path to the unpruned base model")
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--save_init_model', action="store_true", help='save the model after initialization')

# general pruning method related
parser.add_argument('--method', type=str, default="", choices=['', 'L1', 'GReg-1', 'GReg-2', 'Oracle'], 
    help='pruning method name; default is "", implying the original training without any pruning')
parser.add_argument('--stage_pr', type=str, default="", help='to appoint layer-wise pruning ratio')
parser.add_argument('--skip_layers', type=str, default="", help='layer id to skip when pruning')
parser.add_argument('--lr_ft', type=str, default="{0:0.01,30:0.001,60:0.0001,75:0.00001}")
parser.add_argument('--data_path', type=str, default="./data")
parser.add_argument('--wg', type=str, default="filter", choices=['filter', 'channel', 'weight'])
parser.add_argument('--pick_pruned', type=str, default='min', choices=['min', 'max', 'rand'], help='the criterion to select weights to prune')
parser.add_argument('--reinit', type=str, default='', help='before finetuning, the pruned model will be reinited')
parser.add_argument('--not_use_bn', dest='use_bn', default=True, action="store_false", help='if use BN in the network')
parser.add_argument('--block_loss_grad', action="store_true", help="block the grad from loss, only apply weight decay")
parser.add_argument('--save_mag_reg_log', action="store_true", help="save log of L1-norm of filters wrt reg")
parser.add_argument('--save_order_log', action="store_true")
parser.add_argument('--mag_ratio_limit', type=float, default=1000)
parser.add_argument('--base_pr_model', type=str, default=None, help='the model that provides layer-wise pr')
parser.add_argument('--inherit_pruned', type=str, default='index', choices=['index', 'pr'], 
    help='when --base_pr_model is provided, we can choose to inherit the pruned index or only the pruning ratio (pr)')
parser.add_argument('--model_noise_std', type=float, default=0, help='add Gaussian noise to model weights')
parser.add_argument('--model_noise_num', type=int, default=10)
parser.add_argument('--oracle_pruning', action="store_true")
parser.add_argument('--ft_in_oracle_pruning', action="store_true")
parser.add_argument('--last_n_epoch', type=int, default=5, help='in correlation analysis, collect the last_n_epoch loss and average them')
parser.add_argument('--check_jsv_loop', type=int, default=0, help="num of batch loops when checking Jacobian singuar values")

# GReg method related (default setting is for ImageNet):
parser.add_argument('--batch_size_prune', type=int, default=64)
parser.add_argument('--lr_prune', type=float, default=0.001)
parser.add_argument('--update_reg_interval', type=int, default=5)
parser.add_argument('--stabilize_reg_interval', type=int, default=40000)
parser.add_argument('--reg_upper_limit', type=float, default=1.0)
parser.add_argument('--reg_upper_limit_pick', type=float, default=1e-2)
parser.add_argument('--reg_granularity_pick', type=float, default=1e-5)
parser.add_argument('--reg_granularity_prune', type=float, default=1e-4)
parser.add_argument('--reg_granularity_recover', type=float, default=-1e-4)


args = parser.parse_args()
args_tmp = {}
for k, v in args._get_kwargs():
    args_tmp[k] = v

# merge other params (such as method params)
if args.params_json:
    args = merge_args(args, args.params_json)

# Above is the default setting. But if we explicitly assign new value for some arg in the shell script, 
# the following will adjust the arg to the assigned value.
script = " ".join(sys.argv)
for k, v in args_tmp.items():
    if k in script:
        args.__dict__[k] = v

# parse for layer-wise prune ratio
# stage_pr is a list of float, skip_layers is a list of strings
if args.stage_pr:
    if is_single_branch(args.arch): # e.g., alexnet, vgg
        args.stage_pr = parse_prune_ratio_vgg(args.stage_pr, num_layers=num_layers[args.arch]) # example: [0-4:0.5, 5:0.6, 8-10:0.2]
        args.skip_layers = strlist_to_list(args.skip_layers, str) # example: [0, 2, 6]
    else: # e.g., resnet
        args.stage_pr = strlist_to_list(args.stage_pr, float) # example: [0, 0.4, 0.5, 0]
        args.skip_layers = strlist_to_list(args.skip_layers, str) # example: [2.3.1, 3.1]
else:
    assert args.base_pr_model, 'If stage_pr is not provided, base_pr_model must be provided'

# set up lr for finetuning
assert args.lr_ft, 'lr_ft must be provided'
args.lr_ft = strdict_to_dict(args.lr_ft, float)

args.resume_path = check_path(args.resume_path)
args.directly_ft_weights = check_path(args.directly_ft_weights)
args.base_model_path = check_path(args.base_model_path)
args.base_pr_model = check_path(args.base_pr_model)

# some deprecated params to maintain back-compatibility
args.copy_bn_w = True
args.copy_bn_b = True
args.reg_multiplier = 1