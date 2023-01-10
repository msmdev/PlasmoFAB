from Bio import SeqIO
from argparse import ArgumentParser
import numpy as np


def main(args):

    if args.dataset == 'pfal':
        export_pfal_dataset(args, fasta_path='data/pfal/pfal.fasta')
    if args.dataset == 'plasmo_fab':
        export_pfal_dataset(args, fasta_path='data/plasmo_fab/plasmo_fab.fasta')
    elif args.dataset == 'deeploc':
        if args.membrane_only == 'True':
            export_deeploc_dataset(
                args, fasta_path='data/deeploc/deeploc_data.fasta', membrane=True)
        else:
            export_deeploc_dataset(args, fasta_path='data/deeploc/deeploc_data.fasta')


def export_pfal_dataset(args, fasta_path):
    genes, sequences, labels, test = [], [], [], []
    for record in SeqIO.parse(fasta_path, 'fasta'):
        genes.append(str(record.id))
        sequences.append(str(record.seq))
        description = str(record.description).split(' ')
        labels.append(int(description[1]))
        test.append(description[2])

    if args.cut_middle == 'True':
        sequences = __cut_sequences__(sequences=sequences)
    print(genes)
    with open(args.out_file, 'wb') as fh:
        for g, s, l, t in zip(genes, sequences, labels, test):
            if args.length is not None:
                if (len(s) >= int(args.length)):
                    continue
            if args.test == 'True':
                if t == 'False':
                    continue
            header = f">{g}|{l}|{t}\n"
            fh.write(header.encode())
            fh.write(f"{s}\n".encode())
    print('Write OK')
    return


def __cut_sequences__(sequences):
    # cut sequences to 1022 max length because ESM adds two characters
    cut_sequences = []
    for s in sequences:
        #if len(s) > 1022:
        if len(s) > 6000:
            cut_sequences.append(s[:3000] + s[len(s) - 3000:])
            #cut_sequences.append(s[:511] + s[len(s) - 511:])
        else:
            cut_sequences.append(s)
    return cut_sequences


def export_deeploc_dataset(args, fasta_path, membrane=False):
    genes, sequences, bin_labels, ten_state_labels, test = [], [], [], [], []
    for record in SeqIO.parse(fasta_path, "fasta"):
        description = str(record.description).split(' ')
        label = description[1].split('-')
        # do not include samples with unknown label U
        if str(label[1]) == 'U':
            continue
        genes.append(str(record.id))
        sequences.append(str(record.seq))
        test.append(True if len(description) > 2 else False)
        if membrane:
            bin_labels.append(1 if str(label[1]) == 'M' else 0)
        else:
            if (str(label[1]) == 'M') or (str(label[0]) == 'Extracellular'):
                bin_labels.append(1)
            else:
                bin_labels.append(0)
        ten_state_labels.append(str(label[0]))

    if args.cut_middle == 'True':
        sequences = __cut_sequences__(sequences=sequences)

    if args.export_numpy == 'True':
        X_train = [s for s, t in zip(sequences, test) if t == False]
        y_train = [l for l, t in zip(bin_labels, test) if t == False]
        X_test = [s for s, t in zip(sequences, test) if t == True]
        y_test = [s for s, t in zip(bin_labels, test) if t == True]
        print(np.array(X_train).shape)
        print(np.array(X_test).shape)
        np.save('deeploc_X_train.npy', np.array(X_train))
        np.save('deeploc_y_train.npy', np.array(y_train))
        np.save('deeploc_X_test.npy', np.array(X_test))
        np.save('deeploc_y_test.npy', np.array(y_test))

    with open(args.out_file, 'wb') as fh:
        for g, s, l, t in zip(genes, sequences, bin_labels, test):
            if args.length is not None:
                if (len(s) >= int(args.length)):
                    continue
            if args.test == 'True':
                if t == False:
                    continue
            header = f">{g}|{l}|{t}\n"
            fh.write(header.encode())
            fh.write(f"{s}\n".encode())
    print('Write OK')
    return


if __name__ == '__main__':
    parser = ArgumentParser('Parse arguments')
    parser.add_argument('--out_file', help='File to write to')
    parser.add_argument('--dataset', help='which dataset to export')
    parser.add_argument('--test', help='Whether to save only test set')
    parser.add_argument('--membrane_only', help='Whether to export membrane/soluble labels. Only for deeploc.')
    parser.add_argument('--length', help='maximum length to include')
    parser.add_argument('--export_numpy', help='Whether to save train and test numpy files')
    parser.add_argument('--cut_middle', help='Whether to cut out middle part of sequences for ESM embeddings')
    args = parser.parse_args()
    main(args)
