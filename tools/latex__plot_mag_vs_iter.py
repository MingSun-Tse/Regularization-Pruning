import numpy as np
import os
import sys
pjoin = os.path.join
import matplotlib.pyplot as plt

mag_reg_log_dir = sys.argv[1]
net = sys.argv[2]
npys = [x for x in os.listdir(mag_reg_log_dir) if x.endswith('.npy')]
for npy in npys:
    npy = pjoin(mag_reg_log_dir, npy)
    log = np.load(npy).item() # MUST use item, because the loaded npy is actually a dict!
    
    name = log['name']
    name = name.split('module.')[1]
    layer_index = log['layer_index']
    shape = log['shape']
    values = log['values']
    values = np.array(values)
    step = values[:, 0]
    reg  = values[:, 1]
    mag  = values[:, 2]

    picked_steps = [0, 5000, 10000, 15000] if net in ['resnet34', 'resnet50'] else [0, 10000, 20000, 30000]
    for i in range(len(step)):
        s = step[i]
        if s in picked_steps:
            fig, ax = plt.subplots(figsize=(6, 2))
            x = mag[i] / mag[i].max()
            ax.plot(x, color='b', linestyle='-')
            ax.set_ylim([0, 1])

            fs = 8 if net == 'resnet56' else 10
            # ax.legend(prop=dict(size=fs), frameon=False) # fontsize
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
            ax.set_xlabel('Filter index', fontsize=14)
            ax.set_ylabel('Normalized $L_1$-norm', fontsize=14)

            out = pjoin(mag_reg_log_dir, '%s_iter%s_%s_mag.pdf' % (layer_index, s, name))
            fig.savefig(out, bbox_inches='tight')
            plt.close(fig)