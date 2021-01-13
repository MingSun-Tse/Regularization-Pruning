'''
    This code is based on the official PyTorch ImageNet training example 'main.py'. Commit ID: 69d2798, 04/23/2020.
    URL: https://github.com/pytorch/examples/tree/master/imagenet
    Important modified parts will be indicated by '@mst' mark.
    We will do our best to maintain back compatibility.
    Questions to @mingsun-tse (wang.huan@northeastern.edu).
'''
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# --- @mst
import copy
import numpy as np
from importlib import import_module
from data import Data
from logger import Logger
from utils import get_n_params, get_n_flops, get_n_params_, get_n_flops_, PresetLRScheduler, Timer
from utils import add_noise_to_model
from model import model_dict, is_single_branch
from data import num_classes_dict, img_size_dict
from pruner import pruner_dict
from option import args
pjoin = os.path.join

logger = Logger(args)
logprint = logger.log_printer.logprint
accprint = logger.log_printer.accprint
netprint = logger.netprint
timer = Timer(args.epochs)
best_acc1 = 0
best_acc1_epoch = 0
# ---

def main():
    # @mst: move this to above, won't influence the original functions
    # args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1, best_acc1_epoch
    args.gpu = gpu

    if args.gpu is not None:
        logprint("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    num_classes = num_classes_dict[args.dataset]
    img_size = img_size_dict[args.dataset]
    num_channels = 1 if args.dataset == 'mnist' else 3
    if args.dataset in ["imagenet", "imagenet_subset_200"]:
        if args.pretrained:
            logprint("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](num_classes=num_classes, pretrained=True)
        else:
            logprint("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch](num_classes=num_classes)
    else: # @mst: added non-imagenet models
        model = model_dict[args.arch](num_classes=num_classes, num_channels=num_channels, use_bn=args.use_bn)

    # @mst: save the model after initialization if necessary
    if args.save_init_model:
        state = {
                'arch': args.arch,
                'model': model,
                'state_dict': model.state_dict(),
                'ExpID': logger.ExpID,
        }
        save_model(state, mark='init')

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # @mst: load the unpruned model for pruning 
    # This may be useful for the non-imagenet cases where we use our pretrained models
    if args.base_model_path:
        ckpt = torch.load(args.base_model_path)
        if 'model' in ckpt:
            model = ckpt['model']
        model.load_state_dict(ckpt['state_dict'])
        logprint("==> Load pretrained model successfully: '%s'" % args.base_model_path)
        
    # @mst: print base model arch
    netprint(model, 'base model arch')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    # optionally resume from a checkpoint
    # @mst: we will use our option '--resume_path', keep this simply for back-compatibility
    if args.resume:
        if os.path.isfile(args.resume):
            logprint("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logprint("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logprint("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if args.dataset not in ['imagenet', 'imagenet_subset_200']:
        loader = Data(args)
        train_loader = loader.train_loader
        val_loader = loader.test_loader
    else:   
        traindir = os.path.join(args.data_path, args.dataset, 'train')
        folder = 'val3' if args.debug else 'val' # @mst
        valdir = os.path.join(args.data_path, args.dataset, folder)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        test_set = datasets.ImageFolder(valdir, 
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        print('number of test example: %d' % len(test_set))

    # --- @mst: Structured pruning is basically equivalent to providing a new weight initialization before finetune,
    # so just before training, conduct pruning to obtain a new model.
    if args.method:
        if args.dataset in ['imagenet', 'imagenet_subset_200']:
            # imagenet training costs too much time, so we use a smaller batch size for pruning training
            train_loader_prune = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size_prune, shuffle=(train_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        else:
            train_loader_prune = loader.train_loader_prune

        # get the original unpruned model statistics
        # n_params_original = get_n_params(model) # old imple, deprecated 
        # n_flops_original = get_n_flops(model, input_res=img_size, n_channel=num_channels)
        n_params_original_v2 = get_n_params_(model) # test new func, the old one will be removed
        n_flops_original_v2 = get_n_flops_(model, img_size=img_size, n_channel=num_channels) # test new func, the old one will be removed

        prune_state = ''
        if args.resume_path:
            state = torch.load(args.resume_path)
            prune_state = state['prune_state'] # finetune or update_reg or stabilize_reg
            if prune_state == 'finetune':
                model = state['model'].cuda()
                model.load_state_dict(state['state_dict'])
                optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)
                optimizer.load_state_dict(state['optimizer'])
                args.start_epoch = state['epoch']
                logprint("==> Load pretrained model successfully: '{}'. Epoch = {}. prune_state = '{}'".format(
                        args.resume_path, args.start_epoch, prune_state))
        
        if args.wg == 'weight':
            global mask

        if args.directly_ft_weights:
            state = torch.load(args.directly_ft_weights)
            model = state['model'].cuda()
            ''' Will be removed soon!
            if 'model' in state:
                model = state['model'].cuda()
            else: # back-compatible to old saved pth, in which 'model' is not saved. Temporary use, will be removed!
                class passer: pass
                passer.test = validate
                passer.test_loader = val_loader
                passer.train_loader = train_loader_prune
                passer.criterion = criterion
                passer.args = args
                passer.save = save_model
                import pruner.l1_pruner as p
                pruner = p.Pruner(model, args, logger, passer)
                model = pruner.prune()
            '''
            model.load_state_dict(state['state_dict'])
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
            prune_state = 'finetune'
            logprint("==> load pretrained model successfully: '{}'. Epoch = {}. prune_state = '{}'".format(
                    args.directly_ft_weights, args.start_epoch, prune_state))
            if 'mask' in state:
                mask = state['mask']
                apply_mask_forward(model)
                logprint('==> mask restored')

        if prune_state != 'finetune':
            class passer: pass # to pass arguments
            passer.test = validate
            passer.test_loader = val_loader
            passer.train_loader = train_loader_prune
            passer.criterion = criterion
            passer.args = args
            passer.save = save_model
            passer.is_single_branch = is_single_branch
            module = pruner_dict[args.method]

            pruner = module.Pruner(model, args, logger, passer)
            model = pruner.prune() # get the pruned model
            if args.wg == 'weight':
                mask = pruner.mask
                apply_mask_forward(model)
                logprint('==> zero out pruned weight before finetune')

            # since model is new, we need a new optimizer
            if args.arch.startswith('lenet'):
                logprint('==> Using Adam optimizer')
                optimizer = torch.optim.Adam(model.parameters(), args.lr)
            else:
                logprint('==> Using SGD optimizer')
                optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)

        # get the statistics of pruned model
        n_params_now_v2 = get_n_params_(model)
        n_flops_now_v2 = get_n_flops_(model, img_size=img_size, n_channel=num_channels)
        # logprint("==> n_params_original: {:>7.4f}M, n_flops_original: {:>7.4f}G".format(n_params_original, n_flops_original))
        logprint("==> n_params_original_v2: {:>7.4f}M, n_flops_original_v2: {:>7.4f}G".format(n_params_original_v2/1e6, n_flops_original_v2/1e9))
        logprint("==> n_params_now_v2:      {:>7.4f}M, n_flops_now_v2:      {:>7.4f}G".format(n_params_now_v2/1e6, n_flops_now_v2/1e9))
        ratio_param = (n_params_original_v2 - n_params_now_v2) / n_params_original_v2
        ratio_flops = (n_flops_original_v2 - n_flops_now_v2) / n_flops_original_v2
        compression_ratio = 1.0 / (1 - ratio_param)
        speedup_ratio = 1.0 / (1 - ratio_flops)
        logprint("==> reduction ratio -- params: {:>5.2f}% (compression {:>.2f}x), flops: {:>5.2f}% (speedup {:>.2f}x)".format(ratio_param*100, compression_ratio, ratio_flops*100, speedup_ratio))
        
        # test and save just pruned model
        netprint(model, 'model that was just pruned')
        if prune_state != 'finetune':
            t1 = time.time()
            acc1, acc5, loss_test = validate(val_loader, model, criterion, args)
            if args.dataset != 'imagenet': # too costly, not test for now
                acc1_train, acc5_train, loss_train = validate(train_loader, model, criterion, args, noisy_model_ensemble=args.model_noise_std)
            else:
                acc1_train, acc5_train, loss_train = -1, -1, -1
            accprint("Acc1 %.4f Acc5 %.4f Loss_test %.4f | Acc1_train %.4f Acc5_train %.4f Loss_train %.4f | (test_time %.2fs) Just got pruned model, about to finetune" % 
                (acc1, acc5, loss_test, acc1_train, acc5_train, loss_train, time.time()-t1))
            state = {'arch': args.arch,
                    'model': model,
                    'state_dict': model.state_dict(),
                    'acc1': acc1,
                    'acc5': acc5,
                    'ExpID': logger.ExpID,
                    'pruned_wg': pruner.pruned_wg,
                    'kept_wg': pruner.kept_wg,
            }
            if args.wg == 'weight':
                state['mask'] = mask 
            save_model(state, mark="just_finished_prune")

        
        # set lr finetune schduler for finetune
        assert args.lr_ft is not None
        lr_scheduler = PresetLRScheduler(args.lr_ft)
    # ---

    if args.evaluate:
        acc1, acc5, loss_test = validate(val_loader, model, criterion, args)
        logprint('Acc1 %.4f Acc5 %.4f Loss_test %.4f' % (acc1, acc5, loss_test))
        return

    # finetune
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # @mst: use our own lr scheduler
        lr = lr_scheduler(optimizer, epoch) if args.method else adjust_learning_rate(optimizer, epoch, args)
        logprint("==> Set lr = %s @ Epoch %d " % (lr, epoch))

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # @mst: check weights magnitude
        if args.method.endswith("Reg") and 'pruner' in locals():
            for name, m in model.named_modules():
                if name in pruner.reg:
                    ix = pruner.layers[name].layer_index
                    mag_now = m.weight.data.abs().mean()
                    mag_old = pruner.original_w_mag[name]
                    ratio = mag_now / mag_old
                    tmp = '[%2d] %25s -- mag_old = %.4f, mag_now = %.4f (%.2f)' % (ix, name, mag_old, mag_now, ratio)
                    print(tmp, file=logger.logtxt, flush=True)
                    if args.screen_print:
                        print(tmp)

        # evaluate on validation set
        acc1, acc5, loss_test = validate(val_loader, model, criterion, args) # @mst: added acc5
        if args.dataset != 'imagenet': # too costly, not test for now
            acc1_train, acc5_train, loss_train = validate(train_loader, model, criterion, args)
        else:
            acc1_train, acc5_train, loss_train = -1, -1, -1

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_acc1_epoch = epoch
        accprint("Acc1 %.4f Acc5 %.4f Loss_test %.4f | Acc1_train %.4f Acc5_train %.4f Loss_train %.4f | Epoch %d (Best_Acc1 %.4f @ Best_Acc1_Epoch %d) lr %s" % 
            (acc1, acc5, loss_test, acc1_train, acc5_train, loss_train, epoch, best_acc1, best_acc1_epoch, lr))
        logprint('predicted finish time: %s' % timer())

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            if args.method:
                # @mst: use our own save func
                state = {'epoch': epoch + 1,
                        'arch': args.arch,
                        'model': model,
                        'state_dict': model.state_dict(),
                        'acc1': acc1,
                        'acc5': acc5,
                        'optimizer': optimizer.state_dict(),
                        'ExpID': logger.ExpID,
                        'prune_state': 'finetune',
                }
                if args.wg == 'weight':
                    state['mask'] = mask 
                save_model(state, is_best)
            else:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best)
        
def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # @mst: after update, zero out pruned weights
        if args.method and args.wg == 'weight':
            apply_mask_forward(model)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args, noisy_model_ensemble=False):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    train_state = model.training

    # switch to evaluate mode
    model.eval()

    # @mst: add noise to model
    model_ensemble = []
    if noisy_model_ensemble:
        for i in range(args.model_noise_num):
            noisy_model = add_noise_to_model(model, std=args.model_noise_std)
            model_ensemble.append(noisy_model)
        logprint('==> added Gaussian noise to model weights (std=%s, num=%d)' % (args.model_noise_std, args.model_noise_num))
    else:
        model_ensemble.append(model)

    time_compute = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            t1 = time.time()
            output = 0
            for model in model_ensemble: # @mst: test model ensemble
                output += model(images)
            output /= len(model_ensemble)
            time_compute.append((time.time() - t1) / images.size(0))
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     progress.display(i)
            # @mst: commented because of too much log

        # TODO: this should also be done with the ProgressMeter
        # logprint(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))
        # @mst: commented because we will use another print outside 'validate'
    # logprint("time compute: %.4f ms" % (np.mean(time_compute)*1000))

    # change back to original model state if necessary
    if train_state:
        model.train()
    return top1.avg, top5.avg, losses.avg # @mst: added returning top5 acc and loss


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

# @mst: use our own save model function
def save_model(state, is_best=False, mark=''):
    out = pjoin(logger.weights_path, "checkpoint.pth")
    torch.save(state, out)
    if is_best:
        out_best = pjoin(logger.weights_path, "checkpoint_best.pth")
        torch.save(state, out_best)
    if mark:
        out_mark = pjoin(logger.weights_path, "checkpoint_{}.pth".format(mark))
        torch.save(state, out_mark)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# @mst: zero out pruned weights for unstructured pruning
def apply_mask_forward(model):
    global mask
    for name, m in model.named_modules():
        if name in mask:
            m.weight.data.mul_(mask[name])

if __name__ == '__main__':
    main()