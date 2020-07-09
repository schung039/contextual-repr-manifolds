# Emergence of Separable Manifolds in Deep Language Representations

Toolkit for measuring mean-field theoretic manifold analysis (MFTMA) of linguistic manifolds
 implemented
 for the results of the paper [ Emergence of Separable Manifolds in Deep Language Representations ](https://arxiv.org/pdf/2006.01095.pdf), ICML 2020.

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

## Preparing the data
```buildoutcfg
python prepare_data.py -h
usage: prepare_data.py [-h] [--dataset_file DATASET_FILE]
                       [--tag_file TAG_FILE] [--sample SAMPLE]
                       [--max_manifold_size MAX_MANIFOLD_SIZE]
                       [--min_manifold_size MIN_MANIFOLD_SIZE] [--seed SEED]

Prepare data for MFTMA analysis by sampling the dataset such that each
manifold contains between min_manifold_size and max_manifold_size number of
samples.

optional arguments:
  -h, --help            show this help message and exit
  --dataset_file DATASET_FILE
                        Input file with the relevant dataset. Each line
                        contains the space-separated words of the sentence,
                        the tabular character ('\t') and the space-separated
                        respective tags.
  --tag_file TAG_FILE   Input file with the list of tags to use for the MFTMA
                        analysis
  --sample SAMPLE       Output file prefix. This file contains the line index,
                        word index and tag of the randomly sampled dataset.
  --max_manifold_size MAX_MANIFOLD_SIZE
                        The maximal number of words per manifold
  --min_manifold_size MIN_MANIFOLD_SIZE
                        The minimal number of words per manifold
  --seed SEED           Randomization seed.
```
Note that you have to supply dataset file (e.g., Penn Treebank) as input.

## Feature extraction
```buildoutcfg
python feature_extract.py -h
usage: feature_extract.py [-h] [--dataset_file DATASET_FILE]
                          [--tag_file TAG_FILE] [--sample SAMPLE]
                          [--feature_dir FEATURE_DIR]
                          [--pretrained_model_name {bert-base-cased,openai-gpt,distilbert-base-uncased,roberta-base,albert-base-v1}]
                          [--mask] [--random_init]

Extract linguistic features from Transformer.

optional arguments:
  -h, --help            show this help message and exit
  --dataset_file DATASET_FILE
                        Input file with the relevant dataset. Each line
                        contains the space-separated words of the sentence,
                        the tabular character ( ) and the space-separated
                        respective tags.
  --tag_file TAG_FILE   Input file with the list of tags to use for the MFTMA
                        analysis
  --sample SAMPLE       Input file containing the line index, word index and
                        tag of the randomly sampled dataset (output from
                        prepare_data.py.
  --feature_dir FEATURE_DIR
                        Output feature data directory.
  --pretrained_model_name {bert-base-cased,openai-gpt,distilbert-base-uncased,roberta-base,albert-base-v1}
                        Pretrained model name.
  --mask                Boolean indicating whether to mask relevant word.
  --random_init         Boolean indication whether to randomly initialize the
                        model
```

## MFTMA analysis

```buildoutcfg
ython mftma_analysis.py -h
usage: mftma_analysis.py [-h] [--num_layers NUM_LAYERS] [--kappa KAPPA]
                         [--n_t N_T] [--feature_dir FEATURE_DIR]
                         [--mftma_analysis_dir MFTMA_ANALYSIS_DIR]

MFTMA analysis over layers.

optional arguments:
  -h, --help            show this help message and exit
  --num_layers NUM_LAYERS
                        Number of hidden layers.
  --kappa KAPPA         kappa
  --n_t N_T             n_t
  --feature_dir FEATURE_DIR
                        Input feature data directory.
  --mftma_analysis_dir MFTMA_ANALYSIS_DIR
                        Location to output MFTMA analysis directory.
```

## Plotting MFTMA analysis results
```buildoutcfg
python generate_plot.py -h
usage: generate_plot.py [-h] [--output_file_analysis OUTPUT_FILE_ANALYSIS]

Plotting MFTMA analysis results (classification capacity, manifold radius,
manifold dimension and center correlation) over layers

optional arguments:
  -h, --help            show this help message and exit
  --output_file_analysis OUTPUT_FILE_ANALYSIS
                        Location to output analysis data (without file
                        extension)
```