import numpy as np
import os
import sys
pjoin = os.path.join
import matplotlib.pyplot as plt

'''Example:
    python  this_file.py   <mag_reg_log_dir>  <net>
'''


markers = ['*', 'd', 'x', 'o']
colors = ['b', 'r', 'k', 'g', 'orange']
linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
plot_ix_of_this_stage = {
    '0': 0,
    '1': 0,
    '2': 0,
    '3': 0,
    '4': 0
}

def get_std(w_abs):
    out = []
    for x in w_abs:
        out.append(np.std(x) / np.mean(x))
    return np.array(out)

mag_reg_log_dir = sys.argv[1]
net = sys.argv[2]
npys = [x for x in os.listdir(mag_reg_log_dir) if x.endswith('.npy')]
fig, ax = plt.subplots()
for i in range(60): # plot in the order of layer index
    got_it = False
    for npy in npys:
        if npy.startswith("%d_" % i):
            got_it = True
            break
    if got_it == False: continue
    npy = pjoin(mag_reg_log_dir, npy)
    log = np.load(npy).item() # MUST use item, because the loaded npy is actually a dict!
    
    name = log['name']
    name = name.split('module.')[1]
    layer_index = log['layer_index']

    plot_this_layer = False
    if net == 'resnet56':
        # get stage
        stage = int(name.split("layer")[1].split('.')[0])
        key_words = ['.1.conv1', '.3.conv1', '.5.conv1', '.7.conv1'] # the layers that will be plotted
        cond = [kw in name for kw in key_words]
        if any(cond):
            plot_this_layer = True
        label = name
    
    if net == 'vgg19':
        # get stage
        if layer_index < 2:
            stage = 0
        elif layer_index < 4:
            stage = 1
        elif layer_index < 8:
            stage = 2
        elif layer_index < 12:
            stage = 3
        else:
            stage = 4
        
        # 19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
        # choose which layers to plot
        key_words = range(20)
        cond = [kw == layer_index for kw in key_words]
        if any(cond):
            plot_this_layer = True
        label = 'layer %2d' % layer_index
    
    if net == 'resnet50':
        pass
    if net == 'resnet34':
        pass
    if not plot_this_layer: continue

    
    shape = log['shape']
    values = log['values']
    values = np.array(values)
    step = values[:, 0]
    reg = values[:, 1]
    mag = values[:, 2]
    mag_std = get_std(mag)
    plot_ix = plot_ix_of_this_stage[str(stage)]
    ax.plot(reg, mag_std, label=label,
        # marker=markers
        color=colors[stage],
        linestyle=linestyles[plot_ix])
    plot_ix_of_this_stage[str(stage)] += 1

ax.legend()
out = pjoin(mag_reg_log_dir, 'mag_vs_reg.pdf')
fig.savefig(out)
plt.close(fig)
