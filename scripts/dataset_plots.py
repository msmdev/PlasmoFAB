
from utils import load_deeploc_t5_embeddings, load_pfal_t5_embeddings, load_esm_embeddings, load_plasmodb_esm_embeddings

import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.manifold import TSNE
import numpy as np
from joblib import load


def main(dataset, emb, emb_path, fasta, plot, out_file):
    if dataset == 'pfal':
        if emb == 'esm1b':
            X_train, y_train, X_test, y_test = load_esm_embeddings(fasta_path=fasta, emb_path=emb_path)
        elif emb == 'protT5':
            X_train, y_train, X_test, y_test = load_pfal_t5_embeddings(fasta_path=fasta, emb_path=emb_path)
    elif dataset == 'deeploc':
        if emb == 'esm1b':
            X_train, y_train, X_test, y_test = load_esm_embeddings(fasta_path=fasta, emb_path=emb_path)
        elif emb == 'protT5':
            X_train, y_train, X_test, y_test = load_deeploc_t5_embeddings(fasta_path=fasta, emb_path=emb_path)
    elif dataset == 'plasmodb':
        if emb == 'esm1b':
            X = load_plasmodb_esm_embeddings(pfal_fasta=fasta, emb_path=emb_path)
            model = load('pfal_esm1b_lr_model.joblib')
            y = model.predict(X)
            plot_tSNE(x=X, y=y, model='Predicted by pfmp-esm1b-lr', out_file=out_file)
            return
        elif emb == 'protT5':
            return
            # TODO: implement

    if plot == 'tSNE':
        plot_tSNE(
            np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test]),
            model=f"{dataset}_{emb}",
            out_file=out_file)
    #elif plot == 'PCA':
        #plot_PCA(np.concatenate([X_train, y_train]), np.concatenate([y_train, y_test]), model=f"{dataset}_{emb}")



def plot_tSNE(x, y=None, model='', out_file=None):
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(x)

    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, marker='.', cmap=plt.get_cmap('brg'))
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    plt.legend(handles=sc.legend_elements()[0], labels=['non-drug target', 'drug target'], title="Type")
    plt.tight_layout()
    plt.savefig(f"{out_file}.png")


if __name__ == '__main__':
    parser = ArgumentParser('Parse arguments')
    parser.add_argument('--dataset', help='Dataset', choices=['pfal', 'deeploc', 'plasmodb'])
    parser.add_argument('--fasta', help='FASTA of dataset')
    parser.add_argument('--emb_path', help='Embeddings path')
    parser.add_argument('--plot', help='Type of plot', choices=['tSNE', 'PCA'])
    parser.add_argument('--emb', help='Embeeding to use', choices=['esm1b', 'protT5'])
    parser.add_argument('--out_file', help='File to write scores to')
    args = parser.parse_args()
    main(
        dataset=args.dataset,
        fasta=args.fasta,
        emb=args.emb,
        emb_path=args.emb_path,
        plot=args.plot,
        out_file=args.out_file)
