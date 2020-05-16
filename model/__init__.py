from importlib import import_module
import torch
import torch.nn as nn
import copy
import time
import numpy as np
from math import ceil
from collections import OrderedDict

def set_up_model(args, logger):
    logger.log_printer("==> making model ...")
    module = import_module("model.model_%s" % args.method)
    model = module.make_model(args, logger)
    
    if args.resume:
        model.resume()

    if args.pretrained:
        model.load_pretrained()

    return model