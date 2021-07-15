# Regularization-Pruning

This repository is for the new deep neural network pruning methods introduced in the following ICLR 2021 paper:
> **Neural Pruning via Growing Regularization [[Camera Ready](https://openreview.net/pdf?id=o966_Is_nPA)]** \
> [Huan Wang](http://huanwang.tech/), [Can Qin](http://canqin.tech/), [Yulun Zhang](http://yulunzhang.com/), and [Yun Fu](http://www1.ece.neu.edu/~yunfu/) \
> Northeastern University, Boston, MA, USA

TLDR: This paper introduces two new neural network pruning methods (named `GReg-1` and `GReg-2`) based on uniformly growing (L2) regularization:
- `GReg-1` is simply a variant of [magnitude pruning](https://arxiv.org/abs/1608.08710) (i.e., unimportant weights are decided by magnitude sorting). We utilize growing regularition to drive the unimportant weights to zero before evetually removing them.
- `GReg-2` seeks to exploit the Hessian information for more accurate pruning *without Hessian approximation* (which is usually intractable for modern deep nets). The point is that we find with the uniformly growing regularization, how the weights respond can reflect their underlying curvature landscapes, which will ultimately lead to weight seperation in terms of their magnitude (shown in the figure below). When the magnitude gap is large enough, we can faithfully prune them simply by magnitude.
<center><img src="readme_figures/L1norm_vs_iter.png" width="700" hspace="10"></center>

## Step 1: Set up environment
- OS: Linux (Ubuntu 1404 and 1604 checked. It should be all right for most linux platforms. Windows and MacOS not checked.)
- python=3.6.9 (conda to manage environment is *strongly* suggested)
- All the dependant libraries are summarized in `requirements.txt`. Simply install them by `pip install -r requirements.txt`.
- CUDA (We use CUDA 10.2, Driver Version: 440.33.01)

After the installlations, download the code:
```
git clone git@github.com:MingSun-Tse/Regularization-Pruning.git -b master
```

## Step 2: Set up dataset
- We evaluate our methods on CIFAR and ImageNet. CIFAR will be automatically downloaded during training.
- For ImageNet, prepare it following the official [pytorch imagenet example](https://github.com/pytorch/examples/tree/master/imagenet).


## Setp 3: Set up pretrained (unpruned) models
- For ImageNet, we use the official [torchvision models](https://pytorch.org/docs/stable/torchvision/models.html) for fair comparison with other approaches. The pretrained models will be automatically downloaded during training.
- For CIFAR datasets, we train our own models with comparable accuracies with their original papers:
```
# ResNet56, CIFAR10
CUDA_VISIBLE_DEVICES=0 python main.py --arch resnet56 --dataset cifar10 --method L1 --stage_pr [0,0,0,0,0] --batch_size 128 --wd 0.0005 --lr_ft 0:0.1,100:0.01,150:0.001 --epochs 200 --project scratch__resnet56__cifar10

# VGG19, CIFAR100
CUDA_VISIBLE_DEVICES=0 python main.py --arch vgg19 --dataset cifar100 --method L1 --stage_pr [0-18:0] --batch_size 256 --wd 0.0005 --lr_ft 0:0.1,100:0.01,150:0.001 --epochs 200 --project scratch__vgg19__cifar100
```
where `--method` indicates the pruning method; `--stage_pr` is used to indicate the layer-wise pruning ratio (since we train the *unpruned* model here, `stage_pr` is zero. `pr` is short for `pruning_ratio`); `--lr_ft` means learning rate schedule during finetuning.

## Step 4: Training (pruning a pretrained model, not from scratch)

### 1. CIFAR10/100
(1) We use the following snippets to obtain the results on CIFAR10/100 (Table 2 in our paper).
- filter pruning, ResNet56, CIFAR10, speedup=2.55x:
```
# GReg-1
CUDA_VISIBLE_DEVICES=1 python main.py --method GReg-1 -a resnet56 --dataset cifar10 --wd 0.0005 --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 --base_model_path Experiments/*scratch__resnet56__cifar10*/weights/checkpoint_best.pth --batch_size_prune 128 --batch_size 128 --update_reg_interval 10 --stabilize 10000 --stage_pr [0,0.75,0.75,0.32,0] --project GReg-1__resnet56__cifar10__2.55x_pr0.750.32 --screen

# GReg-2
CUDA_VISIBLE_DEVICES=1 python main.py --method GReg-2 -a resnet56 --dataset cifar10 --wd 0.0005 --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 --base_model_path Experiments/*scratch__resnet56__cifar10*/weights/checkpoint_best.pth --batch_size_prune 128 --batch_size 128 --update_reg_interval 10 --stabilize 10000 --stage_pr [0,0.75,0.75,0.32,0] --project GReg-2__resnet56__cifar10__2.55x_pr0.750.32 --screen

```
- filter pruning, VGG19, CIFAR100, speedup=8.84x:
```
# GReg-1
CUDA_VISIBLE_DEVICES=1 python main.py --method GReg-1 -a vgg19 --dataset cifar100 --wd 0.0005 --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 --base_model_path Experiments/*scratch__vgg19__cifar100*/weights/checkpoint_best.pth --batch_size_prune 256 --batch_size 256 --update_reg_interval 10 --stabilize 10000 --stage_pr [1-15:0.7] --project GReg-1__vgg19__cifar100__8.84x_pr0.7 --screen

# GReg-2
CUDA_VISIBLE_DEVICES=1 python main.py --method GReg-2 -a vgg19 --dataset cifar100 --wd 0.0005 --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 --base_model_path Experiments/*scratch__vgg19__cifar100*/weights/checkpoint_best.pth --batch_size_prune 256 --batch_size 256 --update_reg_interval 10 --stabilize 10000 --stage_pr [1-15:0.7] --project GReg-2__vgg19__cifar100__8.84x_pr0.7 --screen
```
> Note: `scratch__resnet56__cifar10`and `*scratch__vgg19__cifar100*` are the experiments of training unpruned models in Step 3.


(2) For the results in Table 1, simply change the pruning ratio using `--stage_pr`:
- ResNet56, CIFAR10: `--stage_pr [0, pr, pr, pr, 0]`, pr in {0.5, 0.7, 0.9, 0.925, 0.95}.
- VGG19, CIFAR100: `--stage_pr [1-15:pr]`, pr in {0.5, 0.6, 0.7, 0.8, 0.9}.

### 2. ImageNet
We use the following snippets to obtain the results on ImageNet (Table 3 and 4 in our paper).

- filter pruning, ResNet34, speedup=1.32x:
```
# GReg-1
CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset imagenet --arch resnet34 --pretrained --method GReg-1 --screen --stage_pr [0,0.5,0.6,0.4,0,0] --skip_layers [1.0,2.0,2.3,3.0,3.5] --project GReg-1__resnet34__imagenet__1.32x_pr0.50.60.4

# GReg-2
CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset imagenet --arch resnet34 --pretrained --method GReg-2 --screen --stage_pr [0,0.5,0.6,0.4,0,0] --skip_layers [1.0,2.0,2.3,3.0,3.5] --project GReg-2__resnet34__imagenet__1.32x_pr0.50.60.4
```

- filter pruning, ResNet50, speedup=1.49x:
```
# GReg-1
CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset imagenet --arch resnet50 --pretrained --method GReg-1 --screen --stage_pr [0,0.3,0.3,0.3,0.14,0] --project GReg-1__resnet50__imagenet__1.49x_pr0.30.14

# GReg-2
CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset imagenet --arch resnet50 --pretrained --method GReg-2 --screen --stage_pr [0,0.3,0.3,0.3,0.14,0] --project GReg-2__resnet50__imagenet__1.49x_pr0.30.14
```

- filter pruning, ResNet50, speedup=2.31x:
```
# GReg-1
CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset imagenet --arch resnet50 --pretrained --method GReg-1 --screen --stage_pr [0,0.6,0.6,0.6,0.21,0] --project GReg-1__resnet50__imagenet__2.31x_pr0.60.21

# GReg-2
CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset imagenet --arch resnet50 --pretrained --method GReg-2 --screen --stage_pr [0,0.6,0.6,0.6,0.21,0] --project GReg-2__resnet50__imagenet__2.31x_pr0.60.21
```

- filter pruning, ResNet50, speedup=2.56x:
```
# GReg-1
CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset imagenet --arch resnet50 --pretrained --method GReg-1 --screen --stage_pr [0,0.74,0.74,0.6,0.21,0] --project GReg-1__resnet50__imagenet__2.56x_pr0.740.60.21

# GReg-2
CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset imagenet --arch resnet50 --pretrained --method GReg-2 --screen --stage_pr [0,0.74,0.74,0.6,0.21,0] --project GReg-2__resnet50__imagenet__2.56x_pr0.740.60.21
```

- filter pruning, ResNet50, speedup=3.06x:

```
# GReg-1
CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset imagenet --arch resnet50 --pretrained --method GReg-1 --screen --stage_pr [0,0.68,0.68,0.68,0.5,0] --project GReg-1__resnet50__imagenet__3.06x_pr0.680.5

# GReg-2
CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset imagenet --arch resnet50 --pretrained --method GReg-2 --screen --stage_pr [0,0.68,0.68,0.68,0.5,0] --project GReg-2__resnet50__imagenet__3.06x_pr0.680.5

```

- unstructured pruning, ResNet50, sparsity=82.7%:
```
# GReg-1
CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset imagenet --arch resnet50 --pretrained --method GReg-1 --wg weight --screen --stage_pr [0,0.827,0.827,0.827,0.827,0.827] --project GReg-1__resnet50__imagenet__wgweight_pr0.827

# GReg-2
CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset imagenet --arch resnet50 --pretrained --method GReg-2 --wg weight --screen --stage_pr [0,0.827,0.827,0.827,0.827,0.827] --project GReg-2__resnet50__imagenet__wgweight_pr0.827
```
where `--wg weight` is to indicate the weight group is weight element, i.e., unstructured pruning.

## ImageNet Results
Our pruned ImageNet models can be downloaded at this [google drive](https://drive.google.com/file/d/1NHq5YSCejYdQyxJYjQWsyfsHgpN2KCtR/view?usp=sharing). Comparison with other methods is shown below. Both structured pruning (filter pruning) and unstructured pruning are evaluated.
> **Tips to load our pruned model**. The pruned model (both the pruned architecture and weights) is saved in the `checkpoint_best.pth`. When loading this file using `torch.load()`, the current path MUST be *the root of this code repository* (because it needs the `model` module in the current directory); otherwise, it will report an error.

(1) Acceleration (structured pruning) comparison on ImageNet
<center><img src="readme_figures/acceleration_comparison_imagenet.png" width="700" hspace="10"></center>

(2) Compression (unstructured pruning) comparison on ImageNet
<center><img src="readme_figures/compression_comparison_imagenet.png" width="700" hspace="10"></center>


## Some useful features
This code also implements some baseline pruning methods that may help you:
- L1-norm pruning. Simple use argument `--method L1`. Example:
```
CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset imagenet --arch resnet50 --pretrained --method L1 --screen --stage_pr [0,0.68,0.68,0.68,0.5,0] --project L1__resnet50__imagenet__3.06x_pr0.680.5
```
- Random pruning and max pruning (i.e., sort the weights by magnitude and prune those with the *largest* magnitudes). There is an argument `--pick_pruned` to decide the sorting criterion. Default is `min`. You may switch to `rand` or `max` for random pruning or max pruning. Example:
```
CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset imagenet --arch resnet50 --pretrained --method L1 --pick_pruned rand --screen --stage_pr [0,0.68,0.68,0.68,0.5,0] --project L1__resnet50__imagenet__3.06x_pr0.680.5__randpruning

CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset imagenet --arch resnet50 --pretrained --method L1 --pick_pruned max --screen --stage_pr [0,0.68,0.68,0.68,0.5,0] --project L1__resnet50__imagenet__3.06x_pr0.680.5__maxpruning
```
- For structured pruning, the pruned network is a compact one, namely, literally *no* zeros stored in the model (so you may find the size of the pruned model is smaller than the original one). For unstructured pruning, we maintain a mask stored in the model (so the model size is not reduced).
- About "--stage_pr" (the argument to adjust layer-wise pruning ratio):
    - For residual networks (i.e., multi-branch), `--stage_pr` is a list. For example, `--stage_pr [0,0.68,0.68,0.68,0.5,0]` means ''stage 0, pr=0; stage 1 to 3, pr=0.68; stage 4, pr=0.5; stage 5, pr=0". FC layer is also counted as the last stage, since we don't prune FC, its pr=0.
    - For vgg-style networks (i.e., single-branch), `--stage_pr` is a dict. For example, `--stage_pr [0-4:0.5, 7-10:0.2]` means "layer 0 to 4, pr=0.5; layer 7-10, pr=0.2; for those not mentioned, pr=0 in default".

Feel free to let us know (raise a GitHub issue or email to `wang.huan@northeastern.edu`. *Email is more recommended if you'd like quicker reply*) if you want any new feature or to evaluate the methods on networks other than those in the paper.


## Acknowledgments
In this code we refer to the following implementations: [pytorch imagenet example](https://github.com/pytorch/examples/tree/master/imagenet), [rethinking-network-pruning](https://github.com/Eric-mingjie/rethinking-network-pruning), [EigenDamage-Pytorch](https://github.com/alecwangcq/EigenDamage-Pytorch), [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10). Great thanks to them!

## Reference
Please cite this in your publication if our work helps your research:

    @inproceedings{wang2021neural,
      Author = {Wang, Huan and Qin, Can and Zhang, Yulun and Fu, Yun},
      Title = {Neural Pruning via Growing Regularization},
      Booktitle = {International Conference on Learning Representations (ICLR)},
      Year = {2021}
    }







