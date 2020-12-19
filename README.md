# Regularization-Pruning

This repository is for the new deep neural network pruning methods introduced in the following paper:
> **Neural Pruning via Growing Regularization [[arxiv](https://arxiv.org/abs/2012.09243)]** \
> [Huan Wang](http://huanwang.tech/), [Can Qin](http://canqin.tech/), [Yulun Zhang](http://yulunzhang.com/), and [Yun Fu](http://www1.ece.neu.edu/~yunfu/) \
> Northeastern University, Boston, MA, USA

TLDR: The paper introduces two new pruning methods (named `GReg-1` and `GReg-2`) based on uniformly growing (L2) regularization.
- `GReg-1` is simply a variant of [magnitude pruning](https://arxiv.org/abs/1608.08710) (i.e., unimportant weights are decided by magnitude sorting). We utilize growing regularition to drive the unimportant weights to zero before evetually removing them.
- `GReg-2` seeks to exploit the Hessian information for more accurate pruning *without Hessian approximation* (which is usually intractable for modern deep nets). The point is that we find with the uniformly growing regularization, how the weights respond can reflect their underlying curvature landscapes, which will ultimately lead to weight seperation in terms of their magnitude. When the magnitude gap is large enough, we can faithfully prune them by magnitude.

## Step 1: Set up environment
- OS: Linux (Ubuntu 1404 and 1604 checked. It should be all right for most linux platforms. Windows and MacOS not checked.)
- python=3.6.9 (conda to manage environment is suggested)
- All the dependant libraries are summarized in `requirements.txt`. Simply install them by `pip install -r requirements`.
- CUDA and cuDNN

After the installlations, download the code:
```
git clone git@github.com:MingSun-Tse/Regularization-Pruning.git -b master
```

## Step 2: Set up dataset
- We evaluate our methods on CIFAR and ImageNet. CIFAR will be automatically downloaded during training.
- For ImageNet, prepare it following the official [pytorch imagenet example](https://github.com/pytorch/examples/tree/master/imagenet).


## Setp 3: Set up pretrained (unpruned) models
- For CIFAR datasets, we train our own models with comparable accuracies with their original papers ([our pretrained CIFAR models](xx)). 
- For ImageNet, we use the official [torchvision models](https://pytorch.org/docs/stable/torchvision/models.html) for fair comparison with other approaches. The pretrained models will be automatically downloaded during training.

## Step 4: Training (pruning)
We use the following snippets to obtain the results on ImageNet (Table 3 and 4 in our paper).
```
# filter pruning, ResNet34, speedup=1.32x, GReg-1
CUDA_VISIBLE_DEVICES=0,1 python main.py --


# filter pruning, ResNet34, speedup=1.32x, GReg-2
CUDA_VISIBLE_DEVICES=0,1 python main.py --


# filter pruning, ResNet34, speedup=1.32x, GReg-2
CUDA_VISIBLE_DEVICES=0,1 python main.py --



# unstructured pruning, ResNet50, sparsity=82.7%, GReg-1

# unstructured pruning, ResNet50, sparsity=82.7%, GReg-2
```




## Results
Our pruned models on ImageNet can be downloaded at this [google drive](xx). Comparison results are shown below. Both structured pruning (filter pruning) and unstructured pruning are evaluated.

(1) Acceleration (structured pruning) comparison on ImageNet
<center><img src="readme_figures/acceleration_comparison_imagenet.png" width="700" hspace="10"></center>

(2) Compression (unstructured pruning) comparison on ImageNet
<center><img src="readme_figures/compression_comparison_imagenet.png" width="700" hspace="10"></center>


**P.S. Tips to load our pruned model**. The pruned model (both the pruned architecture and weights) is saved in the `checkpoint_best.pth`. When loading this file using `torch.load()`, the current path *has to* be in the root of this code repository (because it needs the `model` module in the current dir); otherwise, it will report an error.


## Acknowledgments
In this code we refer to the following implementations: [pytorch imagenet example](https://github.com/pytorch/examples/tree/master/imagenet), [EigenDamage-Pytorch](https://github.com/alecwangcq/EigenDamage-Pytorch), [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10). Great thanks to them!

## Reference
Please cite this in your publication if our work helps your research. Should you have any questions, welcome to reach out to Huan Wang (wang.huan@northeastern.edu).

    @article{wang2020neural,
      Author = {Wang, Huan and Qin, Can and Zhang, Yulun and Fu, Yun},
      Title = {Neural Pruning via Growing Regularization},
      Journal = {arXiv preprint arXiv:2012.09243},
      Year = {2020}
    }







