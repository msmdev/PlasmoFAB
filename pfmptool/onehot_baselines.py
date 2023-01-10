from pfmptool.utils import one_hot_encode, evaluate_model, load_pfal_fasta, load_deeploc_fasta

import os
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef, make_scorer, balanced_accuracy_score, f1_score
import pandas as pd
import numpy as np


class OnehotBaselines():
    """One hot basline models: LR and SVM

    Args:
        PfMPModel (PfMPModel): super class (tbd)
    """

    def __init__(self, randint: int, fasta_path: str, dataset: str):
        """ Initialize model

        Args:
            randint (int): the random integer for K-fold CV
            fasta_path (str): path to data FASTA
            dataset (str): the dataset to use: currently pfal or deeploc

        Raises:
            Exception: If illegal dataset
        """

        if dataset == 'pfal':
            X_train, y_train, X_test, y_test = load_pfal_fasta(
                os.path.join(os.getcwd(), fasta_path))
        elif dataset == 'deeploc':
            X_train, y_train, X_test, y_test = load_deeploc_fasta(
                os.path.join(os.getcwd(), fasta_path), max_length=8000)
        else:
            raise Exception('Unsupported dataset. Only deeploc and pfal supported')
        # pad to longest sequence in either train or test set
        pad_length = max([len(x) for x in np.concatenate([X_train, X_test])])
        self.X_train = one_hot_encode(sequences=X_train, pad_length=pad_length)
        self.y_train = y_train
        self.X_test = one_hot_encode(sequences=X_test, pad_length=pad_length)
        self.y_test = y_test
        self.randint = randint
        self.scorer = {
            'f1': make_scorer(f1_score),
            'mcc': make_scorer(matthews_corrcoef),
            'acc': make_scorer(balanced_accuracy_score)
        }

        self.skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=self.randint)

        indices = []
        for train_idx, test_idx in self.skf.split(self.X_train, self.y_train):
            indices.append([train_idx, test_idx])

    def lr_grid_search(self, params: dict) -> pd.DataFrame:
        """Perform LR grid searc over params.

        Args:
            params (dict): the params for grid search

        Returns:
            pd.DataFrame: Grid CV results
        """
        lr = LogisticRegression(max_iter=4000, solver='liblinear', penalty='l1')
        lr_cv = GridSearchCV(
            lr, params, cv=self.skf, refit='f1',
            scoring=self.scorer, verbose=1, return_train_score=True, n_jobs=-1)
        search = lr_cv.fit(self.X_train, self.y_train)

        lr_res = pd.DataFrame(search.cv_results_).sort_values(by='mean_test_mcc', ascending=False)
        lr_res['mcc_diff'] = lr_res.mean_train_mcc - lr_res.mean_test_mcc

        return lr_res

    def train_lr(self, C: int):
        """Train LR classifier with C.

        Args:
            C (_type_): _description_

        Returns:
            tuple: DataFrame of CV results, test scores
        """
        model = LogisticRegression(max_iter=4000, solver='liblinear', penalty='l1', C=C)
        cv_result = cross_validate(
            model,
            X=self.X_train,
            y=self.y_train,
            cv=self.skf,
            scoring=self.scorer,
            return_train_score=True,
            return_estimator=True,
            verbose=1, n_jobs=-1
        )

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        test_scores = evaluate_model(y_pred=y_pred, y_true=self.y_test)

        return model, pd.DataFrame(cv_result), test_scores

    def rbf_svm_grid_search(self, params: dict) -> pd.DataFrame:
        """_summary_

        Args:
            params (dict): _description_

        Returns:
            pd.DataFrame: Grid CV results
        """
        svm = SVC(kernel='rbf')
        svm_cv = GridSearchCV(
            svm, params, cv=self.skf, refit='f1',
            scoring=self.scorer, verbose=1, return_train_score=True, n_jobs=-1)
        search = svm_cv.fit(self.X_train, self.y_train)

        svm_res = pd.DataFrame(search.cv_results_).sort_values(by='mean_test_mcc', ascending=False)
        svm_res['mcc_diff'] = svm_res.mean_train_mcc - svm_res.mean_test_mcc

        return svm_res

    def train_rbf_svm(self, C: int):
        """Trains SVM RBF classifier with C

        Args:
            C (int): Regularization parameter

        Returns:
            tuple[pd.DataFrame, list]: CV results, test scores
        """
        model = SVC(kernel='rbf', C=C)
        cv_result = cross_validate(
            model,
            X=self.X_train,
            y=self.y_train,
            cv=self.skf,
            scoring=self.scorer,
            return_train_score=True,
            return_estimator=True,
            verbose=1, n_jobs=-1
        )

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        test_scores = evaluate_model(y_pred=y_pred, y_true=self.y_test)

        return model, pd.DataFrame(cv_result), test_scores
