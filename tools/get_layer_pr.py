import numpy as np
import os
import sys
import glob
import math
from collections import OrderedDict
pjoin = os.path.join


'''Usage
    python  this_file.py   '<w_abs log dir>'  <thresh>  Note: the '' is necessary!
    example:
on 138 server:
py get_layer_pr.py ../Experiments/AdaRegPick_resnet50_imagenet_pr0.680.5_bs128_SERVER138-20200522-002904/log/plot 0.1 1
'''
folders = sys.argv[1]
folders = glob.glob(folders)
thresh = float(sys.argv[2])
temp = float(sys.argv[3])

for f in folders:
    pr = {}
    for step in range(1000, 35000, 200):
        npys = [pjoin(f, x) for x in os.listdir(f) if x.endswith('.npy') and ('iter%s_' % step) in x]
        
        for i in npys:
            w_abs_ratio = np.load(i)
            tmp = w_abs_ratio < thresh
            layer = i.split('/')[-1].split('_')[0]
            if layer in pr:
                pr[layer][0] += sum(tmp) / len(w_abs_ratio)
                pr[layer][1] += 1
            else:
                pr[layer] = [sum(tmp) / len(w_abs_ratio), 1]
    
    for k, v in pr.items():
        pr[k] = v[0] / v[1]

    # soften
    values = list(pr.values())
    values = np.array(values)   
    denom = np.exp(values / temp).sum()
    for k, v in pr.items():
        pr[k] = np.exp(v / temp) / denom

    pr_str = '['
    for i in range(100):
        if str(i) in pr:
            pr_str += '%.2f, ' % (pr[str(i)])
    pr_str += ']'
    print(pr_str)

    pr_str = '{'
    for i in range(100):
        if str(i) in pr:
            pr_str += '%s: %.2f, ' % (i, pr[str(i)])
    pr_str += '}'
    print(pr_str)


