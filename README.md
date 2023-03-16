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

### Train models showcased in the manuscript

Pre-computed embeddings (ESM-1b and ProtT5) as well as oligo kernel matrices are available under `data/plasmo_fab`, and training of the logistic regression or SVM can be done locally. `scripts/train.py` is the entry-point to train all models `python train.py -h` prints all available arguments. Model, embedding, regularization parameter, paths, grid search option  and others can be specified.
 
- `scripts/train.py` provides scripts to train the models and can be run from the command line

### Evaluate models showcased in the manuscript

The prediction services which are evaluated in this work all take a FASTA file as input and produce prediction files in various formats (csv, 3line, .txt).

- `scripts/evaluate_results.py` provides functions to parse prediction files for each model and translates their predictions to the binary task of antigen candidate prediction as performed here. Supported models currently are DeepTMHMM, DeepLoc 2.0, DeepLoc 1.0, TMHMM and Phobius.

## How to use PlasmoFAB

If you want to utilize PlasmoFAB, please download the offizial version from Zenodo. Afterwards you can easily load the dataset using your favorite scripting language. For example, loading PlasmoFAB using Python could look like this

```
plasmoFAB_seq = []     # stores the sequences
plasmoFAB_label = []   # stores the labels
plasmoFAB_test = []    # only necessary if you need the test set

with open('PlasmoFAB_pos.csv', 'r') as pos_in:
    next(pos_in)    # skip the header
    for line in pos_in:
        plasmoFAB_seq.append(line.split(',')[1])
        plasmoFAB_label.append(1)
        plasmoFAB_test.append(line.split(',')[2].strip())

with open('PlasmoFAB_neg.csv', 'r') as neg_in:
    next(neg_in)    # skip the header
    for line in neg_in:
        plasmoFAB_seq.append(line.split(',')[1])
        plasmoFAB_label.append(0)
        plasmoFAB_test.append(line.split(',')[2].strip())
```python

To break the sorting, you can simply shuffle the resulting lists

```
import random
plasmoFAB = list(zip(plasmoFAB_seq, plasmoFAB_label, plasmoFAB_test))
random.shuffle(plasmoFAB)
plasmoFAB_seq, plasmoFAB_label, plasmoFAB_test = zip(*plasmoFAB)
```python

Afterwards you can change the sequences into the format needed by your classifier. For example, you could cast the list to numpy arrays and use the *pfmptool.utils.one_hot_encoding* function provided by us to convert the sequences into one-hot-encoded sequences. To get information on how to feed data to your classifier, please consult the API of your framework (e.g. sklearn) or look for tutorials.

## Leaderboard

Here you can find a provisional leaderboard for PlasmoFAB. If you have prediction results of your own model that you wish to be included in this leaderboard, please contact us (see www.pfeiferlab.org for contact details). We are currently looking into the possibility to provide a more standarized leaderboard, so stay tuned.

|Rank         |User         |Model       |MCC         |
|-------------|-------------|------------|------------|
|1            |PfeiferLab   |ProtT5 + LR |0.8500      |
|2            |PfeiferLab   |ProtT5 + SVM|0.8333      |
|2            |PfeiferLab   |ESM1b + LR  |0.8333      |
|2            |PfeiferLab   |ESM1b + SVM |0.8333      |
|5            |PfeiferLab   |DeepTMHMM   |0.7167      |
|6            |PfeiferLab   |DeepLoc 2.0 |0.7009      |
|7            |PfeiferLab   |Oligo-SVM   |0.6500      |
|7            |PfeiferLab   |TMHMM       |0.6500      |
|9            |PfeiferLab   |DeepLoc 1.0 |0.6357      |
|10           |PfeiferLab   |Phobius     |0.6333      |
