import pickle as pkl
import os
import argparse
from mftma.manifold_analysis_correlation import manifold_analysis_corr
import numpy as np

parser = argparse.ArgumentParser(description='MFTMA analysis over layers.')

parser.add_argument('--num_layers', type=int, default=12, help='Number of hidden layers.')
parser.add_argument('--kappa', type=float, default=1e-8, help='kappa')
parser.add_argument('--n_t', type=int, default=1000, help='n_t')

parser.add_argument('--feature_dir', type=str, default='features',
                    help='Input feature data directory.')
parser.add_argument('--mftma_analysis_dir', type=str, default='mftma-analysis',
                    help='Location to output MFTMA analysis directory.')


args = parser.parse_args()
print(args)

print('MFTMA ANALYSIS')

for layer in range(1,args.num_layers+1):

    print('LOADING FEATURE FILES')
    class_encodings = pkl.load(open(os.path.join(args.feature_dir,str(layer)+'.pkl'), 'rb+'))

    # Analyze each layer
    a, r, d, r0, K = manifold_analysis_corr(class_encodings, args.kappa, args.n_t)

    out_data = {'a': a, 'r': r, 'd': d, 'r0': r0, 'K': K}

    # Compute the mean values
    a = 1 / np.mean(1 / a)
    r = np.mean(r)
    d = np.mean(d)
    print("{} capacity: {:4f}, radius {:4f}, dimension {:4f}, correlation {:4f}".format(
        'LAYER_'+str(layer), a, r, d, r0))

    pkl.dump(out_data, open(os.path.join(args.mftma_analysis_dir,str(layer)+'.pkl'), 'wb+'))


print('COMPLETE')