# PlasmoFAB Code Repository

This repository contains scripts that were be used in the creation of the PlasmoFAB dataset. The dataset can be found on [Zenodo](https://doi.org/10.5281/zenodo.7433087). Furthermore, we provide implementations of all experiments described in the manuscript **PlasmoFAB: A Curated Benchmark for Antigen Candidate Exploration in** ***Plasmodium falciparum***.

## Install instructions
-------

1.  Clone the repository  

  `git clone git@github.com:msmdev/PlasmoFAB.git`

2.  Create conda env

`conda env create --name pfmp --file requirements.txt`

3. Run setup.py

`python setup.py install`

## Usage
-------

- `scripts/` provides scripts for most tasks that can be run from command line:

  - `plasmoFAB_dataset.py`: creates the PlasmoFAB dataset from a plasmoDB fasta file when additional files are passed (epitopes, evidence, etc, see script). For details, print help info with `python scripts/pfal_dataset_creation.py --help`
  
  - `train.py`: train a model on specified embedding or run GridSearchCV

  - `boxplot_results.py`: creates the plots when csv files are passed.
 
- `pfmptool/` contains source files for models, loading datasets and other utilties

- `data/` contains data files that were used in the creation of the *PlasmoFAB* version that is described in the manuscript mentioned above. If you want to use *PlasmoFAB*, _always_ use the datafiles provided on Zenodo. This repository was created for reproducibility and is not meant to be used to create your own *PlasmoFAB* version.
 

