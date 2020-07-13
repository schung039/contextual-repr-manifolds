import pickle as pkl
import os
import argparse
from mftma.manifold_analysis_correlation import manifold_analysis_corr

parser = argparse.ArgumentParser(description='MFTMA analysis over layers.')

# Input
parser.add_argument('--feature_dir', type=str, default='features',
                    help='Input feature data directory.')

#Output
parser.add_argument('--mftma_analysis_dir', type=str, default='mftma-analysis',
                    help='Location to output MFTMA analysis directory.')

parser.add_argument('--num_layers', type=int, default=12, help='Number of hidden layers.')

# MFTMA parameters
parser.add_argument('--kappa', type=float, default=1e-8, help='Margin size to use in the '
                                                              'analysis (kappa > 0).')
parser.add_argument('--n_t', type=int, default=1000, help='Number of gaussian vectors to sample '
                                                          'per manifold.')

args = parser.parse_args()
print(args)

for layer in range(1,args.num_layers+1):
    print('MFTMA analysis, layer ' + layer)
    class_encodings = pkl.load(open(os.path.join(args.feature_dir,str(layer)+'.pkl'), 'rb+'))

    a, r, d, r0, K = manifold_analysis_corr(class_encodings, args.kappa, args.n_t)

    mftma_analysis_data = {'a': a, 'r': r, 'd': d, 'r0': r0, 'K': K}
    pkl.dump(mftma_analysis_data, open(os.path.join(args.mftma_analysis_dir, str(layer) + '.pkl'), 'wb+'))