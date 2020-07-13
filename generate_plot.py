import os
import pickle as pkl
import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Plotting MFTMA analysis results (classification '
                                             'capacity, manifold radius, manifold dimension '
                                             'and center correlation) over layers')
parser.add_argument('--num_layers', type=int, default=12, help='Number of hidden layers.')
parser.add_argument('--mftma_analysis_dir', type=str, default='mftma-analysis',
                    help='Location to output MFTMA analysis directory.')

args = parser.parse_args()
print(args)

capacities = []
radii = []
dimensions = []
correlations = []

for layer in range(1,args.num_layers+1):
    temp_data = pkl.load(open(os.path.join(args.mftma_analysis_dir,str(layer)+'.pkl'), 'rb+'))
    a = 1 / np.mean(1 / temp_data['a'])
    r = np.mean(temp_data['r'])
    d = np.mean(temp_data['d'])
    r0 = temp_data['r0']
    if layer == 1:
        norm_a = a
        norm_r = r
        norm_d = d
        norm_r0 = r0

    a /= norm_a
    r /= norm_r
    d /= norm_d
    r0 /= norm_r0
    print("{} capacity: {:4f}, radius {:4f}, dimension {:4f}, correlation {:4f}".format(
        'LAYER_' + str(layer), a, r, d, r0))

    capacities.append(a)
    radii.append(r)
    dimensions.append(d)
    correlations.append(r0)

fig, axes = plt.subplots(1, 4, figsize=(18, 4))

axes[0].plot(capacities, linewidth=5)
axes[1].plot(radii, linewidth=5)
axes[2].plot(dimensions, linewidth=5)
axes[3].plot(correlations, linewidth=5)

axes[0].set_ylabel(r'$\alpha_M$', fontsize=18)
axes[1].set_ylabel(r'$R_M$', fontsize=18)
axes[2].set_ylabel(r'$D_M$', fontsize=18)
axes[3].set_ylabel(r'$\rho_{center}$', fontsize=18)

xticklabels = [i for i in range(1,args.num_layers+1)]
for ax in axes:
    ax.set_xticks([i for i, _ in enumerate(xticklabels)])
    ax.set_xlabel('Layer')
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()
plt.show()