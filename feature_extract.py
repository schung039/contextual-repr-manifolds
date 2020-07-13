import os
from collections import defaultdict
import pickle as pkl
import argparse
import numpy as np

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel

parser = argparse.ArgumentParser(description='Extract linguistic features from Transformer.')

# Input
parser.add_argument('--dataset_file', type=str, default="dataset/ptb_pos.txt",
                    help='Input file with the relevant dataset. Each line contains the '
                         'space-separated words of the sentence,  the tabular character (\t) and '
                         'the space-separated respective tags.')
parser.add_argument('--tag_file', type=str,
                    default="dataset/relevant_pos.txt",
                    help='Input file with the list of tags to use for the MFTMA analysis/')
parser.add_argument('--sample', type=str,
                    default="dataset/sample_seed_0.pkl",
                    help='Input file containing the line index, '
                         'word index and tag of the randomly sampled dataset (output from '
                         'prepare_data.py.')

# Output
parser.add_argument('--feature_dir', type=str, default='features',
                    help='Output feature data directory.')

# Parameters
parser.add_argument('--pretrained_model_name', type=str, default='bert-base-cased',
                    choices=['bert-base-cased', 'openai-gpt', 'distilbert-base-uncased',
                             'roberta-base', 'albert-base-v1'], help='Pretrained model name.')
parser.add_argument('--mask', action='store_true', default=False,
                    help='Boolean indicating whether to mask relevant word.')
parser.add_argument('--random_init', action='store_true', default=False,
                    help='Boolean indication whether to randomly initialize the model.')


args = parser.parse_args()
print(args)

print('Extracting Features')

tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
config = AutoConfig.from_pretrained(args.pretrained_model_name, output_hidden_states=True)
if args.random_init: # random initialization of the model
    model = AutoModel.from_config(config)
else:
    model = AutoModel.from_pretrained(args.pretrained_model_name, config=config)

manifold_vectors = defaultdict(dict)
with open(args.tag_file) as f:
    for tag in f:
        tag = tag.strip().lower()
        for layer in range(1,config.num_hidden_layers+1):
            manifold_vectors[layer][tag] = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

line_word_tag_map = pkl.load(open(args.sample, 'rb+'))

with open(args.dataset_file, encoding='utf-8') as dfile:
    for line_idx,line in enumerate(dfile):
        if line_idx in line_word_tag_map:
            words, tags = line.strip().split('\t')
            word_list = list(words.split())
            for word_idx in line_word_tag_map[line_idx]:
                tag = line_word_tag_map[line_idx][word_idx].lower()
                if args.mask:
                    # replace the word_idx location with mask token
                    word_list[word_idx] = tokenizer.mask_token

                if args.pretrained_model_name == 'openai-gpt':
                    split_word_idx = []
                else:
                    split_word_idx = [-1]

                # tokenization - assign the same id for all sub words of a same word
                word_tokens = []
                for split_id, split_word in enumerate(word_list):
                    tokens = tokenizer.tokenize(split_word)
                    word_tokens.extend(tokens)
                    split_word_idx.extend([split_id] * len(tokens))

                if args.pretrained_model_name != 'openai-gpt':
                    split_word_idx.append(len(word_list))

                input_ids = torch.Tensor([tokenizer.encode(word_tokens, add_special_tokens=True)]).long()
                input_ids = input_ids.to(device)
                with torch.no_grad():
                    model_output = model(input_ids)[-1]
                for layer in range(1,config.num_hidden_layers+1):
                    layer_output = model_output[layer][0]
                    vector_idcs = np.argwhere(np.array(split_word_idx) == word_idx).reshape(-1)
                    token_vector = layer_output[vector_idcs].mean(0).cpu().reshape(-1,1).numpy()
                    if manifold_vectors[layer][tag] is None:
                        manifold_vectors[layer][tag] = token_vector
                    else:
                        manifold_vectors[layer][tag] = np.hstack((manifold_vectors[layer][tag],
                                                                token_vector))

for layer in range(1,config.num_hidden_layers+1):
    pkl.dump(list(manifold_vectors[layer].values()), open(os.path.join(args.feature_dir,
                                                                  str(layer)+'.pkl'), 'wb+'))