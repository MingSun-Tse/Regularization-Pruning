# Regularization-Pruning

This repository is for the GReg pruning method introduced in the following paper: \
Neural Pruning via Growing Regularization [[Arxiv](https://arxiv.org/abs/2012.09243)] \
[Huan Wang](http://huanwang.tech/), [Can Qin](http://canqin.tech/), [Yulun Zhang](http://yulunzhang.com/), and [Yun Fu](http://www1.ece.neu.edu/~yunfu/) \
Northeastern University, Boston, MA, USA

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







