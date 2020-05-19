import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math

'''Usgae:
    python  this_file.py  <path to log.txt>
'''
f = sys.argv[1]
mode = sys.argv[2]
mag_ratio = {}
lines = [line.strip() for line in open(f)]
n_line = len(lines)
for i in range(n_line):
    line = lines[i]
    if "Mag ratio = " in line:
        k = 1
        while 'reg for' not in lines[i-k]:
            k += 1
        if mode == 'old':
            step = lines[i-k].split('iter = ')[1].split(')')[0]
        else:
            step = lines[i-k].split('Iter = ')[1].split('.')[0]
        step = int(step)
        layer_ix = lines[i-k].split(' ')[3].split('[')[1].split(']')[0]
        mag_r = min(float(line.split('(')[1].split(')')[0]), 1e5)
        if layer_ix in mag_ratio:
            mag_ratio[layer_ix].append([step, mag_r])
        else:
            mag_ratio[layer_ix] = [[step, mag_r]]

for k, v in mag_ratio.items():
    fig, ax = plt.subplots()
    v = np.array(v)
    ax.plot(v[:,0], v[:,1])
    ax.set_title(k)
    ax.set_ylim([0, 1e5])
    out_path = f.replace('log.txt', '%s.jpg' % k)
    fig.savefig(out_path)
    plt.close(fig)
