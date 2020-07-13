import random
import pickle as pkl
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description='Prepare data for MFTMA analysis by sampling the '
                                             'dataset such that each manifold '
                                             'contains between min_manifold_size and '
                                             'max_manifold_size number of samples.')

# Input
parser.add_argument('--dataset_file', type=str, default="dataset/ptb_pos.txt",
                    help='Input file with the relevant dataset. Each line contains the '
                         'space-separated words of the sentence,  the tabular character (\\t) and '
                         'the space-separated respective tags.')
parser.add_argument('--tag_file', type=str,
                    default="dataset/relevant_pos.txt",
                    help='Input file with the list of tags to use for the MFTMA analysis/')

# Output
parser.add_argument('--sample', type=str, default="dataset/sample_seed_0.pkl",
                    help='Output file containing the  line index, '
                         'word index and tag of the randomly sampled dataset.')

# Parameters
parser.add_argument('--max_manifold_size', type=int, default=50,
                    help='The maximal number of words per manifold.')
parser.add_argument('--min_manifold_size', type=int, default=1,
                    help='The minimal number of words per manifold.')
parser.add_argument('--seed', type=int, default=0,
                    help='Randomization seed.')

args = parser.parse_args()

print(args)

random.seed(args.seed)

relevant_tags = set()
with open(args.tag_file) as f:
    for tag in f:
        relevant_tags.add(tag.strip().lower())

# open dataset file and get tags and words - save them to a dictionary
tag2location = defaultdict(list)
with open(args.dataset_file) as f:
    for line_idx, line in enumerate(f):
        tags = line.strip().split('\t')[1].lower().split()
        for word_idx, tag in enumerate(tags):
            if tag in relevant_tags:
                tag2location[tag].append((line_idx, word_idx))

word_count = 0
line_word_tag_map = defaultdict(dict)
for tag in tag2location:
    if len(tag2location[tag]) > args.max_manifold_size:
        locations = random.sample(tag2location[tag], args.max_manifold_size)
    elif len(tag2location[tag]) >= args.min_manifold_size:
        locations = tag2location[tag]
    else:
        continue
    for location in locations:
        line_idx, word_idx = location
        word_count += 1
        line_word_tag_map[line_idx][word_idx] = tag

print('Number of Words', word_count)
print('Number of Manifolds', len(tag2location))
pkl.dump(line_word_tag_map, open(args.sample,'wb+'))