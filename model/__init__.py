from importlib import import_module
from .vgg import vgg11, vgg13, vgg16, vgg19
from .resnet_cifar10 import resnet56
from .lenet5 import lenet5

def set_up_model(args, logger):
    logger.log_printer("==> making model ...")
    module = import_module("model.model_%s" % args.method)
    model = module.make_model(args, logger)    
    return model

model_dict = {
    'lenet5': lenet5,
    'resnet56': resnet56,
    'vgg11': vgg11,
    'vgg13': vgg13,
    'vgg16': vgg16,
    'vgg19': vgg19,
}

num_layers = {
    'lenet5': 5,
    'alexnet': 8,
    'vgg11': 11,
    'vgg13': 13,
    'vgg16': 16,
    'vgg19': 19,
}