import joblib
from argparse import ArgumentParser
import os
import sys
sys.path.append('../')
from pfmptool.pfmp_esm1b import PfMPESM1b
from pfmptool.onehot_baselines import OnehotBaselines
from pfmptool.pfmp_protT5 import PfMPProtT5
from pfmptool.oligo_svm import OligoSVM


def main():

    randint = 38173

    parser = ArgumentParser('Parse arguments')
    parser.add_argument('--model', help='Model to train')
    parser.add_argument(
        '--grid_search',
        help='Whether to perform grid search for parameter optimization',
        type=bool,
        default=False)
    parser.add_argument(
        '--embedding', help='Embedding to use', choices=['protT5', 'esm1b', 'onehot', 'oligo'])
    parser.add_argument('--C', help='Regularization parameter C for SVM and LR', default='0.1')
    parser.add_argument('--dataset', help='Dataset to use', choices=['plasmoFAB', 'pfal', 'deeploc', 'plasmodb'])
    parser.add_argument('--kernel_path', help='Path to oligo kernel .npy files', default=None)
    parser.add_argument('--fasta', help='Path to dataset FASTA file', default=None)
    parser.add_argument('--emb_path', help='Path to embeddings directory')
    parser.add_argument(
        '--grid_C', help='List of comma-separated C floats used in grid search', default=None)
    parser.add_argument(
        '--out_file', help='File path to write results (of grid search or scores) to.', default=None
    )
    parser.add_argument('--save_csv', help='Whether to save result csv to file')
    parser.add_argument('--train_final', help='Whether to save trained clf to pickle', default=None)

    args = parser.parse_args()

    if args.out_file is None:
        raise Exception("Provide out_file parameter")

    if args.grid_C is not None:
        params = dict(C=[float(c) for c in args.grid_C.split(',')])
    if (args.grid_C is None) & (args.grid_search):
        raise Exception("Provide grid_C parameter for grid search")

    fasta_path = args.fasta

    model = None
    if args.embedding == 'onehot':
        model = OnehotBaselines(randint=randint, fasta_path=fasta_path, dataset=args.dataset)
    elif args.embedding == 'protT5':
        if args.emb_path is None:
            raise Exception(f"Provide embedding path for {args.embedding} model")
        model = PfMPProtT5(
            randint=randint,
            fasta_path=fasta_path,
            emb_path=args.emb_path,
            dataset=args.dataset)
    elif args.embedding == 'esm1b':
        if args.emb_path is None:
            raise Exception(f"Provide embedding path for {args.embedding} model")
        model = PfMPESM1b(
            randint=randint,
            fasta_path=fasta_path,
            emb_path=args.emb_path,
            dataset=args.dataset)
    elif args.embedding == 'oligo':
        model = OligoSVM(
            fasta_path=fasta_path,
            dataset=args.dataset,
            k_mer_length=1, sigma=18,
            randint=randint,
            kernel_path=args.kernel_path)
    else:
        raise Exception('Embedding not supported.')

    grid_results, cv_results, test_scores = None, None, None
    if args.model == 'lr':
        if args.grid_search == True:
            grid_results = model.lr_grid_search(params=params)
        else:
            estimator, cv_results, test_scores = model.train_lr(C=float(args.C))
    elif args.model == 'svm':
        if args.grid_search:
            grid_results = model.rbf_svm_grid_search(params=params)
        else:
            estimator, cv_results, test_scores = model.train_rbf_svm(C=float(args.C))
    elif args.model == 'oligo_svm':
        if args.grid_search:
            grid_results = model.oligo_svm_grid_search(params=params)
        else:
            estimator, cv_results, test_scores = model.train_svm(C=float(args.C))
    else:
        raise Exception('Model not supported. Possible types are lr, svm or mlp.')

    try:
        os.remove(os.path.join(os.getcwd(), args.out_file + '_testscores.txt'))
    except OSError:
        pass
    try:
        os.remove(os.path.join(os.getcwd(), args.out_file + '_grid.txt'))
    except OSError:
        pass

    if args.out_file is not None:
        if args.grid_search:
            grid_results.to_csv(f"{args.out_file}_grid.csv")
            with open(args.out_file + '_grid.txt', 'a') as f:
                result_str = grid_results[
                ['param_C', 'mean_test_mcc', 'mean_train_mcc', 'mcc_diff',
                    'mean_test_f1', 'mean_train_f1', 'mean_test_acc']].to_string(
                    header=True, index=True)
                f.write(result_str)
        else:
            cv_results.to_csv(f"{args.out_file}_cv.csv")
            joblib.dump(estimator, f"{args.out_file}.joblib")
            with open(args.out_file + '_testscores.txt', 'a') as f:
                f.write(f"Model {str(args.model)} embedding {str(args.embedding)} C = {float(args.C)}\n")
                f.write('------------------------------------------\n')
                f.write('------------------------------------------\n')
                f.write(f"MCC:           {round(test_scores[0], 4)}\n")
                f.write(f"f1:            {round(test_scores[1], 4)}\n")
                f.write(f"Accuracy:      {round(test_scores[2], 4)}\n")


if __name__ == '__main__':
    main()
