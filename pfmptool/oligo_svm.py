# run like
# python3 scripts/train.py --embedding oligo --model oligo_svm --out_file pfal_oligo_svm --dataset pfal --kernel data/pfal/kernels
# grid_search = True for grid search

import os

from pfmptool.utils import load_pfal_fasta, load_deeploc_fasta, evaluate_cv_models, evaluate_model

#from pyBeast.OligoKernel import oligoKernel
#from pyBeast.OligoEncoding import oligoEncoding

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, f1_score, make_scorer
from sklearn.svm import SVC


class OligoSVM():

    def __init__(
        self,
        fasta_path,
        dataset,
        randint,
        k_mer_length=1,
        sigma=18,
        kernel_path=None,
    ):

        self.k_mer_length = k_mer_length
        self.sigma = sigma
        self.fasta_path = fasta_path
        self.randint = randint
        self.dataset = dataset

        if self.dataset == 'pfal':
            self.X_train, self.y_train, self.X_test, self.y_test = load_pfal_fasta(fasta_path)
        elif self.dataset == 'deeploc':
            self.X_train, self.y_train, self.X_test, self.y_test = load_deeploc_fasta(fasta_path)
        else:
            raise Exception("Unsupported dataset. Only pfal and deeploc supported")

        self.scorer = {
            'f1': make_scorer(f1_score),
            'mcc': make_scorer(matthews_corrcoef),
            'acc': make_scorer(balanced_accuracy_score)
        }

        self.skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=self.randint)

        self.skf_indices = []
        for train_idx, test_idx in self.skf.split(self.X_train, self.y_train):
            self.skf_indices.append([train_idx, test_idx])

        if kernel_path is not None:
            self.train_kernel = np.load(os.path.join(
                #kernel_path, f"pfal_train_kernel_kmer{k_mer_length}_sigma{sigma}.npy"))
                kernel_path, f"{str(self.dataset)}_train_kernel_kmer_{k_mer_length}_sigma_{sigma}.npy"))
            self.test_kernel = np.load(os.path.join(
                #kernel_path, f"pfal_test_kernel_kmer{k_mer_length}_sigma{sigma}.npy"))
                kernel_path, f"{str(self.dataset)}_test_kernel_kmer_{k_mer_length}_sigma_{sigma}.npy"))
        else:
            raise Exception('Computing kernels not yet supported.')
            #if out_path is None:
            #    save_path = f"{os.getcwd()}/data/{self.dataset}/oligoKernels/"
            #else:
            #    save_path = out_path
            #positions, values = oligoEncoding(
            #    k_mer_length=int(k_mer_length),
            #    alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ').getEncoding(self.X_train)
            #pos_test, val_test = oligoEncoding(
            #    k_mer_length=int(k_mer_length),
            #    alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ').getEncoding(self.X_test)

            #self.ftrain_kernel = oligoKernel(
            #    sigma=sigma, max_distance=True).symmetric(positions, values)
            #self.test_kernel = oligoKernel(
            #    sigma=sigma, max_distance=True).asymmetric(pos_test, val_test, positions, values)
            #np.save(
            #    f"{save_path}/train_kernel_kmer_{k_mer_length}_sigma_{sigma}", self.train_kernel)
            #np.save(
            #    f"{save_path}/test_kernel_kmer_{k_mer_length}_sigma_{sigma}", self.test_kernel)

    def oligo_svm_grid_search(
        self, params=dict(C=[0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.003, 0.005])
    ):
        # kmer = 1 : sigma = 18, c = 0.0005  <-- best performing
        # kmer = 2 : sigma = 16, c = 0.0005 | sigma = 10, c = 0.001
        # kmer = 3 : sigma = 16, c = 0.001  | sigma = 12, c = 0.0005
        clf = SVC(kernel='precomputed')
        cv = GridSearchCV(
            clf, params, cv=self.skf, refit='mcc',
            scoring=self.scorer, verbose=1, return_train_score=True, n_jobs=-1)
        search = cv.fit(self.train_kernel, self.y_train)

        res = pd.DataFrame(search.cv_results_).sort_values(by='mean_test_mcc', ascending=False)
        res['mcc_diff'] = res.mean_train_mcc - res.mean_test_mcc

        return res

    def train_svm2(self, C=0.0005):
        clf = SVC(kernel='precomputed', C=C)
        cv_result = cross_validate(
            clf,
            X=self.train_kernel,
            y=self.y_train,
            cv=self.skf,
            scoring=self.scorer,
            return_train_score=True,
            return_estimator=True,
            verbose=1, n_jobs=-1
        )

        y_preds = [cv_model.predict(
            self.test_kernel[:, self.skf_indices[i][0]]) for i, cv_model in enumerate(cv_result['estimator'])]
        test_scores = evaluate_cv_models(
            estimators=cv_result['estimator'], y_preds=y_preds, y_true=self.y_test)

        return pd.DataFrame(cv_result), test_scores

    def train_svm(self, C):
        model = SVC(kernel='precomputed', C=C)
        cv_result = cross_validate(
            model,
            X=self.train_kernel,
            y=self.y_train,
            cv=self.skf,
            scoring=self.scorer,
            return_train_score=True,
            return_estimator=True,
            verbose=1, n_jobs=-1
        )

        model.fit(self.train_kernel, self.y_train)
        y_pred = model.predict(self.test_kernel)
        test_scores = evaluate_model(y_pred=y_pred, y_true=self.y_test)

        return model, pd.DataFrame(cv_result), test_scores
