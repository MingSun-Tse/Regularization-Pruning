import numpy as np
import os
import sys
pjoin = os.path.join
import matplotlib.pyplot as plt

'''Example:
    python  this_file.py   <mag_reg_log_dir>  <net>
'''


markers = ['*', 'd', 'x', 'o']
colors = ['b', 'r', 'k', 'g']
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
fig, ax = plt.subplots(figsize=(6,5))
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
    if net in ['resnet56', 'resnet34']: # non-bottleneck block
        # get stage
        stage = int(name.split("layer")[1].split('.')[0])

        # choose which layers to plot
        key_words = ['.0.conv1', '.2.conv1', '.4.conv1', '.6.conv1'] # the layers that will be plotted
        cond = [kw in name for kw in key_words]
        if any(cond):
            plot_this_layer = True
        label = name
    
    if net == 'resnet50': # bottleneck block
        # get stage
        stage = int(name.split("layer")[1].split('.')[0])

        # choose which layers to plot
        key_words = ['.0.conv', '.2.conv'] # the layers that will be plotted
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

    if not plot_this_layer: continue
    shape = log['shape']
    values = log['values']
    values = np.array(values)
    step = values[:, 0][::3]
    reg  = values[:, 1][::3]
    mag  = values[:, 2][::3]
    mag_std = get_std(mag)
    plot_ix = plot_ix_of_this_stage[str(stage)]
    ax.plot(reg, mag_std, label=label,
        # marker=markers
        color=colors[stage - 1], # start with the blue color
        linestyle=linestyles[plot_ix],
        linewidth=1.5)
    plot_ix_of_this_stage[str(stage)] += 1

fs = 8 if net == 'resnet56' else 10
ax.legend(prop=dict(size=fs), frameon=False) # fontsize
ax.grid(color='white')
ax.set_facecolor('whitesmoke')

# remove axis line
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# remove tick but keep the values
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# set x ylabel
ax.set_xlabel('Regularization factor $\lambda$', fontsize=14)
ax.set_ylabel('Normalized $L_1$-norm stddev', fontsize=14)

out = pjoin(mag_reg_log_dir, 'mag_vs_reg.pdf')
fig.savefig(out, bbox_inches='tight')
plt.close(fig)