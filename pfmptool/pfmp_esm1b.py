from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import make_scorer, balanced_accuracy_score, matthews_corrcoef, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
import numpy as np

from pfmptool.utils import evaluate_model, load_esm_embeddings


class PfMPESM1b():

    def __init__(self, randint, fasta_path, emb_path, dataset):

        self.emb_path = emb_path

        self.fasta_path = fasta_path

        # load embeddings
        self.X_train, self.y_train, self.X_test, self.y_test = load_esm_embeddings(emb_path=self.emb_path)

        self.randint = randint
        self.scorer = {
            'f1': make_scorer(f1_score),
            'mcc': make_scorer(matthews_corrcoef),
            'acc': make_scorer(balanced_accuracy_score)
        }

        self.skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=self.randint)
        print("Dataset ", self.X_train.shape)

    def lr_grid_search(self, params=dict(C=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])):

        lr = LogisticRegression(max_iter=4000, solver='liblinear', penalty='l2')
        lr_cv = GridSearchCV(
            lr, params, cv=self.skf, refit='f1',
            scoring=self.scorer, verbose=1, return_train_score=True, n_jobs=-1)
        search = lr_cv.fit(self.X_train, self.y_train)

        lr_res = pd.DataFrame(search.cv_results_).sort_values(by='mean_test_mcc', ascending=False)
        lr_res['mcc_diff'] = lr_res.mean_train_mcc - lr_res.mean_test_mcc

        return lr_res

    def train_lr(self, C=0.1):

        model = LogisticRegression(max_iter=4000, solver='liblinear', penalty='l2', C=C)
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

    def rbf_svm_grid_search(self, params=dict(C=[1, 5, 10, 15, 20, 25, 30])):

        svm = SVC(kernel='rbf')
        svm_cv = GridSearchCV(
            svm, params, cv=self.skf, refit='f1',
            scoring=self.scorer, verbose=1, return_train_score=True, n_jobs=-1)
        search = svm_cv.fit(self.X_train, self.y_train)

        lr_res = pd.DataFrame(search.cv_results_).sort_values(by='mean_test_mcc', ascending=False)
        lr_res['mcc_diff'] = lr_res.mean_train_mcc - lr_res.mean_test_mcc

        return lr_res

    def train_rbf_svm(self, C=1):
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

    def train_final(self, C=0.2):
        lr = LogisticRegression(max_iter=4000, solver='liblinear', penalty='l2', C=C)
        X = np.concatenate([self.X_train, self.X_test])
        y = np.concatenate([self.y_train, self.y_test])
        return lr.fit(X, y)
