import sys
import os
import numpy as np 
import matplotlib.pyplot as plt
plt.style.use(['science', 'no-latex'])

def get_set_diff(x, y):
    diff = [i for i in x if i not in y]
    return len(diff) * 1.0 / len(x)

def check_order_stability(order):
    order = np.array(order)
    n_item = len(order)
    order_change = order[-n_item+1:] - order[-n_item:-1]
    ave_order_change = np.abs(order_change).mean(axis=1)
    return ave_order_change

def check_picked_wg_stability(wg_preprune):
    wg_preprune = np.array(wg_preprune)
    n_item = len(wg_preprune)
    ratio_picked_wg_change = [get_set_diff(wg_preprune[i + 1], wg_preprune[i]) for i in range(n_item - 1)]
    return np.array(ratio_picked_wg_change)

# ------------------------------------
inFile = sys.argv[1]
# ------------------------------------
order_by_L1 = {}
wg_preprune = {}
layer_index = {}
for line in open(inFile):
    splits = line.split()
    layer_name = splits[3]
    layer_index[layer_name] = splits[2][6:] # example: Layer#11
    ranking = [int(x) for x in splits[5:]]

    if splits[4] == 'order_by_L1':
        if layer_name in order_by_L1:
            order_by_L1[layer_name].append(ranking)
        else:
            order_by_L1[layer_name] = [ranking]
    else:
        if layer_name in wg_preprune:
            wg_preprune[layer_name].append(ranking)
        else:
            wg_preprune[layer_name] = [ranking]

print('==> parsing done')
for layer_name in order_by_L1.keys():
    print('==> plotting layer %s' % layer_name)
    ave_order_change = check_order_stability(order_by_L1[layer_name])
    ratio_picked_wg_change = check_picked_wg_stability(wg_preprune[layer_name])
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    axes[0].plot(ave_order_change, label='ave_order_change')
    axes[1].plot(ratio_picked_wg_change, label='ratio_picked_wg_change')
    axes[0].legend()
    axes[1].legend()
    out_path = inFile.replace('.txt', '_order_analysis_layer%s.jpg' % layer_index[layer_name])
    fig.savefig(out_path, bbox_inches='tight')