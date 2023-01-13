# PlasmoFAB Code Repository

This repository contains scripts that were be used in the creation of the PlasmoFAB dataset. The dataset can be found on [Zenodo](https://doi.org/10.5281/zenodo.7433087). Furthermore, we provide implementations of all experiments described in the manuscript **PlasmoFAB: A Benchmark to Foster Machine Learning for** ***Plasmodium falciparum*** **Protein Antigen Candidate Prediction**.


## Install instructions
-------

1.  Clone the repository  

  `git clone git@github.com:msmdev/PlasmoFAB.git`

2.  Create conda env

`conda env create --name pfmp --file requirements.txt`

3. Run setup.py

`python setup.py install`

 
## Reproduce dataset

**WARNING**: If you want to use *PlasmoFAB* for your own experiments, **always** use the datafiles provided on Zenodo. This repository was created for reproducibility and is not meant to be used to create your own *PlasmoFAB* version.

`scripts/plasmoFAB_dataset.py` contains all pre-processing and data collection steps to produce the final plasmoFAB dataset from various input files (see `data/plasmoFAB/data_sources`. 

The file paths are hardcoded in a dictionary but will work when this repo is cloned. Note that using differing input files will lead to a possible different outcome.

## Train models

Pre-computed embeddings (ESM-1b and ProtT5) as well as oligo kernel matrices are available under `data/plasmo_fab`, and training of the logistic regression or SVM can be done locally. `scripts/train.py` is the entry-point to train all models `python train.py -h` prints all available arguments. Model, embedding, regularization parameter, paths, grid search option  and others can be specified.
 
- `scripts/` provides scripts for most tasks that can be run from command line:

## Evaluate models

The prediction services which are evaluated in this work all take a FASTA file as input and produce prediction files in various formats (csv, 3line, .txt).

`scripts/evaluate_results.py` provides functions to parse prediction files for each model and translates their predictions to the binary task of antigen candidate prediction as performed here. Supported models currently are DeepTMHMM, DeepLoc 2.0, DeepLoc 1.0, TMHMM and Phobius.
