#
# python3 scripts/train.py --embedding protT5 --model lr --grid_search True  --out_file lr_t5.txt --dataset deeploc
#
from pfmptool.utils import load_t5_embeddings, evaluate_model


from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import make_scorer, balanced_accuracy_score, matthews_corrcoef, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd


class PfMPProtT5():
    """PfMP ProtT5 class

    Args:
        PfMPModel (class): super class
    """

    def __init__(self, randint: int, fasta_path: str, emb_path: str, dataset: str):
        """Initialize model

        Args:
            randint (int): random int for CV
            fasta_path (str): data fasta
            emb_path (str): embedding path
            dataset (str): dataset: pfal or deeploc
        """

        self.emb_path = emb_path
        self.fasta_path = fasta_path
        self.dataset = dataset

        self.X_train, self.y_train, self.X_test, self.y_test = load_t5_embeddings(
            fasta_path=self.fasta_path, emb_path=self.emb_path, dataset=self.dataset)

        self.randint = randint
        self.scorer = {
            'f1': make_scorer(f1_score),
            'mcc': make_scorer(matthews_corrcoef),
            'acc': make_scorer(balanced_accuracy_score)
        }

        self.skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=self.randint)

    def train_cv(self, model: str, params: dict):
        if model == 'lr':
            model = LogisticRegression(max_iter=4000, solver='liblinear', penalty='l2')
        elif model == 'SVM':
            model = SVC(kernel='rbf')
        else:
            raise Exception("Only models 'lr' and 'SVM' supported for ProtT5")
        cv = GridSearchCV(model, params, cv=self.skf, refit='mcc',
            scoring=self.scorer, verbose=1, return_train_score=True, n_jobs=-1)

        search = cv.fit(self.X_train, self.y_train)
        estimator = search.best_estimator_
        res = pd.DataFrame(search.cv_results_).sort_values(by='mean_test_mcc', ascending=False)
        res['mcc_diff'] = res.mean_train_mcc - res.mean_test_mcc

        y_pred = estimator.predict(self.X_test)
        test_scores = evaluate_model(y_pred=y_pred, y_true=self.y_test)
        print("test scores ", test_scores)
        return estimator, res

    def lr_grid_search(self, params=dict(C=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])):

        lr = LogisticRegression(max_iter=4000, solver='liblinear', penalty='l2')
        # params = dict(C=[0.06, 0.08, 0.1])
        lr_cv = GridSearchCV(
            lr, params, cv=self.skf, refit='mcc',
            scoring=self.scorer, verbose=1, return_train_score=True, n_jobs=-1)
        search = lr_cv.fit(self.X_train, self.y_train)

        lr_res = pd.DataFrame(search.cv_results_).sort_values(by='mean_test_mcc', ascending=False)
        lr_res['mcc_diff'] = lr_res.mean_train_mcc - lr_res.mean_test_mcc

        return lr_res

    def train_f(self, model, C=0.01):
        if model == 'lr':
            model = LogisticRegression(max_iter=4000, solver='liblinear', penalty='l2', C=C)
        elif model == 'SVM':
            model = SVC(kernel='rbf', C=C)
        else:
            raise Exception("Only models 'lr' and 'SVM' supported for ProtT5")
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)
        test_scores = evaluate_model(y_pred=y_pred, y_true=self.y_test, model=f"ProtT5-{model}, C = {C}")

        return model, test_scores

    def train_lr(self, C=0.01):
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

    def rbf_svm_grid_search(self, params=dict(C=[0.8, 1, 1.2, 1.4, 1.6, 1.8, 2])):

        svm = SVC(kernel='rbf')
        svm_cv = GridSearchCV(
            svm, params, cv=self.skf, refit='mcc',
            scoring=self.scorer, verbose=1, return_train_score=True, n_jobs=-1)
        search = svm_cv.fit(self.X_train, self.y_train)

        lr_res = pd.DataFrame(search.cv_results_).sort_values(by='mean_test_mcc', ascending=False)
        lr_res['mcc_diff'] = lr_res.mean_train_mcc - lr_res.mean_test_mcc

        return lr_res

    def train_rbf_svm(self, C):
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
