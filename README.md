# Regularization-Pruning

This repository is for the deep neural network pruning method introduced in the following paper:
> Neural Pruning via Growing Regularization [[Arxiv](https://arxiv.org/abs/2012.09243)] \
> [Huan Wang](http://huanwang.tech/), [Can Qin](http://canqin.tech/), [Yulun Zhang](http://yulunzhang.com/), and [Yun Fu](http://www1.ece.neu.edu/~yunfu/) \
> Northeastern University, Boston, MA, USA

TLDR: The paper introduces two new pruning methods (named `GReg-1` and `GReg-2`) based on uniformly growing (L2) regularization.
- `GReg-1` is simply a variant of [magnitude pruning](https://arxiv.org/abs/1608.08710) (i.e., unimportant weights are decided by magnitude sorting). We utilize growing regularition to drive the unimportant weights to zero before evetually removing them.
- `GReg-2` seeks to exploit the Hessian information for more accurate pruning *without Hessian approximation* (which is usually intractable for modern deep nets). The point is that we find with the uniformly growing regularization, how the weights respond can reflect their underlying curvature landscapes, which will ultimately lead to weight seperation in terms of their magnitude. When the magnitude gap is large enough, we can faithfully prune them by magnitude.

## Step 1: Set up environment
- OS: Linux (Ubuntu 1404 and 1604 checked. It should be all right for most linux platforms. Windows and MacOS not checked.)
- python==3.6.9 (conda to manage environment is suggested)
- All the dependant libraries are summarized in `requirements.txt`. Simply install them by `pip install -r requirements`.
- CUDA and cuDNN

After the installlations, download the code:
```
git clone git@github.com:MingSun-Tse/Regularization-Pruning.git -b master
```

## Step 2: Set up dataset
- We evaluate our methods on CIFAR and ImageNet. CIFAR will be automatically downloaded.
- For ImageNet, prepare it following the official [pytorch imagenet example](https://github.com/pytorch/examples/tree/master/imagenet).


## Setp 3: Set up pretrained (unpruned) models
- For CIFAR datasets, we train our own models with comparable accuracies with their original papers ([our pretrained CIFAR models](xx)). 
- For ImageNet, we use the official [torchvision models](https://pytorch.org/docs/stable/torchvision/models.html) for fair comparison with others. They will be automatically downloaded during training.

## Step 4: Run (pruning)


## Results
Our pruned models on ImageNet can be downloaded at this [google drive](xx). Comparison results are shown below.
<center><img src="readme_figures/acceleration_comparison_imagenet.png" width="400" hspace="10"></center>


## Acknowledgments
In this code we refer to the following implementations: [pytorch imagenet example](https://github.com/pytorch/examples/tree/master/imagenet), [EigenDamage-Pytorch](https://github.com/alecwangcq/EigenDamage-Pytorch), [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10). Great thanks to them!

## Reference
Please cite this in your publication if our work helps your research. Should you have any questions, welcome to reach out to Huan Wang (wang.huan@northeastern.edu).

    @article{wang2020neural,
      Author = {Wang, Huan and Qin, Can and Zhang, Yulun and Fu, Yun},
      Title = {Neural Pruning via Growing Regularization},
      Journal = {},
      Year = {2020}
    }







