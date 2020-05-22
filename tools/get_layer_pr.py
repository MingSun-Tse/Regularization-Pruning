import numpy as np
import os
import sys
pjoin = os.path.join

'''Usage
    python  this_file.py   <w_abs log dir>   <step>   <thresh>
'''
folder = sys.argv[1]
step = sys.argv[2]
thresh = float(sys.argv[3])

npys = [pjoin(folder, x) for x in os.listdir(folder) if x.endswith('.npy') and ('iter%s_' % step) in x]
pr = {}
for i in npys:
    w_abs_ratio = np.load(i)
    tmp = w_abs_ratio < thresh
    layer = i.split('/')[-1].split('_')[0]
    pr[layer] = sum(tmp) / len(w_abs_ratio)

pr_str = '['
for i in range(100):
    if str(i) in pr:
        pr_str += '%.2f, ' % (pr[str(i)])
pr_str += ']'
print(pr_str)


