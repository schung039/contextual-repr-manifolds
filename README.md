# Emergence of Separable Manifolds in Deep Language Representations

Toolkit for measuring mean-field theoretic manifold analysis (MFTMA) of linguistic manifolds
 implemented in Python
 for the results of the paper [Emergence of Separable Manifolds in Deep Language Representations](https://arxiv.org/pdf/2006.01095.pdf), ICML 2020.


## Install

First install required dependencies with
```
pip install -r requirements.txt
```

Then install the package via
```
pip install -e .
```
## Usage
The following contains usage instructions for preparing the data, extracting the features, run
 MFTMA analysis and plot the results.

### Preparing the data
Prepare the data for MFTMA analysis given a labeled dataset (`dataset_file`), the list of
 relevant tags (`tag_file`) and a
 randomization seed (`seed`).
The dataset should be formatted such that each line contains the space-separated words of the
 sentence, the tabular character ('\t') and the space-separated respective tags.
Here is an example of a line from [Penn Treebank](https://catalog.ldc.upenn.edu/desc/addenda/LDC99T42.pos.txt
), \
`So how many , um , credit cards do you have ? \t UH WRB JJ , UH , NN NNS VBP PRP VB .`\
The script samples the dataset such that each
manifold contains between `min_manifold_size` and `max_manifold_size` number of
samples. It produces 'sample' file that contains the line index, word index and tag of the
 randomly sampled dataset for MFTMA analysis.

```buildoutcfg
usage: prepare_data.py [-h] [--dataset_file DATASET_FILE]
                       [--tag_file TAG_FILE] [--sample SAMPLE]
                       [--max_manifold_size MAX_MANIFOLD_SIZE]
                       [--min_manifold_size MIN_MANIFOLD_SIZE] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_file DATASET_FILE
                        Input file with the relevant dataset. Each line
                        contains the space-separated words of the sentence,
                        the tabular character (\t) and the space-separated
                        respective tags.
  --tag_file TAG_FILE   Input file with the list of tags to use for the MFTMA
                        analysis/
  --sample SAMPLE       Output file containing the line index, word index and
                        tag of the randomly sampled dataset.
  --max_manifold_size MAX_MANIFOLD_SIZE
                        The maximal number of words per manifold.
  --min_manifold_size MIN_MANIFOLD_SIZE
                        The minimal number of words per manifold.
  --seed SEED           Randomization seed.
```

### Feature extraction
Extract linguistic features from Transformer given the `dataset_file`, `tag_file`, the name
 of the pretrained transformer model `pretrained_model_name`. Use `mask` for masking tokens and
  `random_init` for random initialization of the pretrained model. We use
 [ HuggingFace
 Transformers library](https://github.com/huggingface/transformers).
Extracted features are stored under `feature_dir`.

```buildoutcfg
usage: feature_extract.py [-h] [--dataset_file DATASET_FILE]
                          [--tag_file TAG_FILE] [--sample SAMPLE]
                          [--feature_dir FEATURE_DIR]
                          [--pretrained_model_name {bert-base-cased,openai-gpt,distilbert-base-uncased,roberta-base,albert-base-v1}]
                          [--mask] [--random_init]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_file DATASET_FILE
                        Input file with the relevant dataset. Each line
                        contains the space-separated words of the sentence,
                        the tabular character ( ) and the space-separated
                        respective tags.
  --tag_file TAG_FILE   Input file with the list of tags to use for the MFTMA
                        analysis/
  --sample SAMPLE       Input file containing the line index, word index and
                        tag of the randomly sampled dataset (output from
                        prepare_data.py.
  --feature_dir FEATURE_DIR
                        Output feature data directory.
  --pretrained_model_name {bert-base-cased,openai-gpt,distilbert-base-uncased,roberta-base,albert-base-v1}
                        Pretrained model name.
  --mask                Boolean indicating whether to mask relevant word.
  --random_init         Boolean indication whether to randomly initialize the
                        model.
```

### MFTMA analysis
MFTMA analysis of the extracted features over layers, given the number of layers `num_layers`, .
the margin size to use in the analysis `kappa` and the number of gaussian vectors to sample per
 manifold `n_t`.
 
We use a python implementation of the MFTMA analysis method developed available
 at [Replica Mean Field Theory Analysis of Object Manifolds](https://github.com/schung039/neural_manifolds_replicaMFT/).
Relevant code has been copied under [mftma](https://github.com/schung039/contextual-repr-manifolds/tree/master/mftma).

MFTMA analysis output is stored under `mftma_analysis_dir` folder.

```buildoutcfg
usage: mftma_analysis.py [-h] [--feature_dir FEATURE_DIR]
                         [--mftma_analysis_dir MFTMA_ANALYSIS_DIR]
                         [--num_layers NUM_LAYERS] [--kappa KAPPA] [--n_t N_T]

MFTMA analysis over layers.

optional arguments:
  -h, --help            show this help message and exit
  --feature_dir FEATURE_DIR
                        Input feature data directory.
  --mftma_analysis_dir MFTMA_ANALYSIS_DIR
                        Location to output MFTMA analysis directory.
  --num_layers NUM_LAYERS
                        Number of hidden layers.
  --kappa KAPPA         Margin size to use in the analysis (kappa > 0).
  --n_t N_T             Number of gaussian vectors to sample per manifold.
```

### Plotting MFTMA analysis results
Plotting MFTMA analysis results (classification capacity, manifold radius,
manifold dimension and center correlation) over layers given the location of the folder with
 MFTMA analysis `mftma_analysis_dir`.
```buildoutcfg
usage: generate_plot.py [-h] [--num_layers NUM_LAYERS]
                        [--mftma_analysis_dir MFTMA_ANALYSIS_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --num_layers NUM_LAYERS
                        Number of hidden layers.
  --mftma_analysis_dir MFTMA_ANALYSIS_DIR
                        Location to output MFTMA analysis directory.
```

## Reference
If you find this code useful for your research, please cite [our paper](https://arxiv.org/pdf/2006.01095.pdf):
```
@InProceedings{mamou2020emergence,
  title={Emergence of Separable Manifolds in Deep Language Representations},
  author={Mamou, Jonathan and Le, Hang and Del Rio, Miguel and Stephenson, Cory and Tang, Hanlin and Kim, Yoon and Chung, SueYeon},
  booktitle={37th International Conference on Machine Learning, ICML 2020},
  year={2020},
  organization={International Machine Learning Society (IMLS)}
}
```
