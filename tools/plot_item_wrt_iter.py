import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math

'''Usgae:
This file is to plot a certain item for each layer w.r.t. iter during training.
That item can be 'mag ratio', 'order diff', etc.
    python  plot_item_wrt_iter.py  <path to log.txt>  new  'Order diff:'
'''
# ------------------------------------------------
mode_choices = ['old', 'new']
item_choices = ['Mag ratio =', 'Order diff:']
f    = sys.argv[1]
mode = sys.argv[2] # used to parse 'Iter'
item = sys.argv[3]
assert mode in mode_choices
assert item in item_choices
# ------------------------------------------------
item_to_plot = {}
lines = [line.strip() for line in open(f)]
n_line = len(lines)
for i in range(n_line):
    line = lines[i]
    if item in line:
        
        # parse Iter
        k = 1
        while 'Update reg for layer' not in lines[i-k]:
            k += 1
        if mode == 'old':
            step = lines[i-k].split('iter = ')[1].split(')')[0]
        else:
            step = lines[i-k].split('Iter = ')[1].split('.')[0]
        step = int(step)
        
        # get layer index
        layer_ix = lines[i-k].split(' ')[3].split('[')[1].split(']')[0]
        
        # get item value
        if item == 'Mag ratio =':
            value = min(float(line.split('(')[1].split(')')[0]), 1e5)
        else:
            value = float(line.split(item)[1].strip().split()[0])
        
        if layer_ix in item_to_plot:
            item_to_plot[layer_ix].append([step, value])
        else:
            item_to_plot[layer_ix] = [[step, value]]

for k, v in item_to_plot.items():
    fig, ax = plt.subplots()
    v = np.array(v)
    ax.plot(v[:,0], v[:,1])
    ax.set_title(k)
    # ax.set_ylim([0, 1e5])
    out_path = f.replace('log.txt', '%s_%s.jpg' % (item, k))
    fig.savefig(out_path)
    plt.close(fig)
