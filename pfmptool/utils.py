import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, f1_score, roc_auc_score
import os
import torch
from Bio import SeqIO


def pfal_fasta_to_dataframe(fasta: str) -> pd.DataFrame:
    """Creates DataFrame from fasta

    Args:
        fasta (str): pfal data fasta

    Returns:
        pd.DataFrame: pfal as Dataframe
    """
    df = pd.DataFrame(columns=['transcript', 'seq', 'type', 'test'])
    for record in SeqIO.parse(fasta, 'fasta'):
        transcript = str(record.id)
        seq = str(record.seq)
        description = str(record.description).split(' ')
        label = int(description[1])
        test = True if str(description[2]) == 'True' else False
        df.loc[len(df.index)] = [transcript, seq, label, test]
    return df


def load_pfal_fasta(path_to_fasta: str):
    """Loads pfal fasta and returns train, test splits

    Args:
        path_to_fasta (str): path to fasta

    Returns:
        tuple[np.array, np.array, np.array, np.array]: X_train, y_train, X_test, y_test
    """
    genes, sequences, labels, test = [], [], [], []
    for record in SeqIO.parse(path_to_fasta, 'fasta'):
        genes.append(str(record.id))
        sequences.append(str(record.seq))
        description = str(record.description).split(' ')
        labels.append(int(description[1]))
        test.append(description[2])

    X_train = [s for s, t in zip(sequences, test) if t == 'False']
    y_train = [s for s, t in zip(labels, test) if t == 'False']

    X_test = [s for s, t in zip(sequences, test) if t == 'True']
    y_test = [s for s, t in zip(labels, test) if t == 'True']

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def load_t5_embeddings(
        fasta_path: str,
        emb_path: str,
        dataset: str):
    """Load T5 embeddings and return as train, test

    Args:
        fasta_path (str): fasta path
        emb_path (str): embedding path
        dataset (str): dataset: currently pfal or deeploc

    Raises:
        Exception: if illegal dataset

    Returns:
        tuple[np.array, np.array, np.array, np.array]: X_train, y_train, X_test, y_test
    """

    if dataset == 'pfal':
        df = pfal_fasta_to_dataframe(fasta_path)
    elif dataset == 'deeploc':
        df = deeploc_fasta_to_dataframe(fasta_path)
    elif dataset == 'plasmo_fab':
        df = pfal_fasta_to_dataframe(fasta_path)
    else:
        raise Exception("Unsupported dataset. Only pfal and deeploc supported")
    # locate train and test samples
    g_train = df.loc[df.test == False]['transcript']
    g_test = df.loc[df.test == True]['transcript']
    y_train = np.array(df.loc[df.test == False]['type'])
    y_test = np.array(df.loc[df.test == True]['type'])

    X_train = np.array(
        [np.load(os.path.join(emb_path, f) + '.npy').squeeze(axis=0) for f in g_train], dtype=float)
    X_test = np.array(
        [np.load(os.path.join(emb_path, f) + '.npy').squeeze(axis=0) for f in g_test], dtype=float)

    return X_train, y_train, X_test, y_test


def load_esm_embeddings(
        emb_path: str,
        layer: int = 33):
    """Load ESM embeddings

    Args:
        emb_path (str): path to embeddings
        layer (int, optional): layer (only 33 available). Defaults to 33.

    Returns:
        tuple[np.array, np.array, np.array, np.array]: _description_
    """

    files = os.listdir(emb_path)
    genes, labels, test, embs = [], [], [], []
    for file in files:
        emb = torch.load(os.path.join(emb_path, file), map_location=torch.device("cpu"))
        sample = file.removesuffix('.pt').split('|')
        genes.append(sample[0])
        labels.append(int(sample[1]))
        test.append(True if sample[2] == 'True' else False)
        embs.append(emb['mean_representations'][layer])
    emb_train = [x for x, t in zip(embs, test) if t == False]
    emb_test = [x for x, t in zip(embs, test) if t == True]
    X_train = torch.stack(emb_train, dim=0).numpy()
    X_test = torch.stack(emb_test, dim=0).numpy()

    y_train = [x for x, t in zip(labels, test) if t == False]
    y_test = [x for x, t in zip(labels, test) if t == True]

    return X_train, y_train, X_test, y_test


def load_plasmodb_esm_embeddings(emb_path: str, pfal_fasta: str, layer: int = 33) -> np.array:
    """Load embeddings of complete plasmodb data

    Args:
        emb_path (str): embedding path
        pfal_fasta (str): fasta
        layer (int, optional): layer: only 33. Defaults to 33.

    Returns:
        np.array: array of embeddings
    """
    pfal = pfal_fasta_to_dataframe(pfal_fasta)
    files = os.listdir(emb_path)
    genes, embs = [], []
    for file in files:
        sample = file.removesuffix('.pt')
        # load only samples not contained in pfal set
        if sample in pfal.transcript.unique():
            continue
        emb = torch.load(os.path.join(emb_path, file), map_location=torch.device("cpu"))
        genes.append(sample[0])
        embs.append(emb['mean_representations'][layer])
    X = torch.stack(embs, dim=0).numpy()
    return X


def plasmodb_fasta_to_dataframe(fasta_path: str) -> pd.DataFrame:
    """Load plasmoDB dataframe

    Args:
        fasta_path (str): fasta

    Returns:
        pd.DataFrame: plasmoDB data
    """
    df = pd.DataFrame(columns=[
        'id', 'gene', 'transcript', 'gene_description', 'description', 'length', 'pseudo', 'seq']
    )
    with open(fasta_path, 'r') as file:
        for rec in SeqIO.parse(file, 'fasta'):
            description = str(rec.description).split('|')
            gene = description[2].strip().split('=')[1].strip()
            transcript = description[1].split('=')[1].strip()
            gene_description = description[4].split('=')[1].strip()
            pseudo = description[10].split('=')[1].strip()
            description = description[5].split('=')[1].strip()
            df.loc[len(df.index)] = [str(rec.id), gene, transcript, gene_description, description,
                                     len(str(rec.seq)), pseudo, str(rec.seq)]
    return df


def pfal_dataframe_to_fasta(df: pd.DataFrame, file: str) -> None:
    """Write pfal DataFrame to fasta

    Args:
        df (pd.DataFrame): pfal dataframe
        file (str): file path
    """
    with open(file, 'wb') as fh:
        for row in df.itertuples():
            header = f">{getattr(row, 'transcript')} {getattr(row, 'type')} {getattr(row, 'test')}\n"
            fh.write(header.encode())
            fh.write(f"{getattr(row, 'seq')}\n".encode())
    print('Write OK')
    return


def deeploc_fasta_to_dataframe(fasta_path: str) -> pd.DataFrame:
    """Load deeploc to dataframe.

    Args:
        fasta_path (str): fasta path

    Returns:
        pd.DataFrame: deeploc dataframe
    """
    df = pd.DataFrame(columns=[
        'transcript', 'seq', 'type', 'ten_state_type', 'test']
    )
    for record in SeqIO.parse(fasta_path, "fasta"):
        description = str(record.description).split(' ')
        label = description[1].split('-')
        # do not include samples with unknown label U
        if str(label[1]) == 'U':
            continue
        gene = str(record.id)
        sequence = str(record.seq)
        test = True if len(description) > 2 else False
        if (str(label[1]) == 'M') or (str(label[0]) == 'Extracellular'):
            bin_label = 1
        else:
            bin_label = 0
        ten_state_label = str(label[0])
        df.loc[len(df.index)] = [gene, sequence, bin_label, ten_state_label, test]
    return df


def deeploc_membrane_fasta_to_dataframe(fasta_path):
    df = pd.DataFrame(columns=[
        'transcript', 'seq', 'type', 'ten_state_type', 'test']
    )
    for record in SeqIO.parse(fasta_path, "fasta"):
        description = str(record.description).split(' ')
        label = description[1].split('-')
        # do not include samples with unknown label U
        if str(label[1]) == 'U':
            continue
        gene = str(record.id)
        sequence = str(record.seq)
        test = True if len(description) > 2 else False
        if (str(label[1]) == 'M'):
            bin_label = 1
        else:
            bin_label = 0
        ten_state_label = str(label[0])
        df.loc[len(df.index)] = [gene, sequence, bin_label, ten_state_label, test]
    return df


def load_deeploc_fasta(fasta_path: str, max_length: int = -1):
    """_summary_

    Args:
        fasta_path (str): _description_
        max_length (int, optional): _description_. Defaults to -1.

    Returns:
        tuple: X_train, y_train, X_test, y_test
    """
    genes, sequences, bin_labels, ten_state_labels, test = [], [], [], [], []
    for record in SeqIO.parse(fasta_path, "fasta"):
        description = str(record.description).split(' ')
        label = description[1].split('-')
        # do not include samples with unknown label U
        if str(label[1]) == 'U':
            continue
        if (max_length > 0) & (len(str(record.seq)) > max_length):
            continue
        genes.append(str(record.id))
        sequences.append(str(record.seq))
        test.append(True if len(description) > 2 else False)
        if (str(label[1]) == 'M') or (str(label[0]) == 'Extracellular'):
            bin_labels.append(1)
        else:
            bin_labels.append(0)
        ten_state_labels.append(str(label[0]))
    X_train = [s for s, t in zip(sequences, test) if t == False]
    y_train = [s for s, t in zip(bin_labels, test) if t == False]

    X_test = [s for s, t in zip(sequences, test) if t == True]
    y_test = [s for s, t in zip(bin_labels, test) if t == True]

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def one_hot_encode(sequences: np.array, pad_length: int = 0) -> np.array:
    """One hot encode sequences

    Args:
        sequences (np.array): sequences
        pad_length (int, optional): padding length. Defaults to 0.

    Returns:
        np.array: one-hot encoded sequences
    """

    alphabet = 'ABCDEFGHIKLMNPQRSTUVWXYZ_'
    if pad_length == 0:
        pad_length = max([len(x) for x in sequences])
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    # int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    encodings = np.zeros(shape=(len(sequences), pad_length, len(alphabet)))
    for i, seq in enumerate(sequences):
        seq = seq.ljust(pad_length, '_')
        seq_encoding = np.zeros(shape=(pad_length, len(alphabet)))
        for j, aa in enumerate(seq):
            aa_onehot = np.zeros(len(alphabet))
            aa_onehot[char_to_int[aa]] = 1
            seq_encoding[j] = aa_onehot
        encodings[i] = seq_encoding
    return encodings.reshape(len(sequences), -1)


def evaluate_cv_models(estimators: list, y_preds: list, y_true: np.array) -> pd.DataFrame:
    """Evaluate predition of CV models and return scores

    Args:
        estimators (list): the estimators
        y_preds (list): predictions, list of lists
        y_true (np.array): true types, same shape as y_preds

    Returns:
        pd.DataFrame: _description_
    """
    df = pd.DataFrame(columns=['estimator', 'test_mcc', 'test_f1', 'test_acc'])
    for est, y_pred in zip(estimators, y_preds):
        mcc = matthews_corrcoef(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        acc = balanced_accuracy_score(y_true, y_pred)
        df.loc[len(df.index)] = [est, mcc, f1, acc]
    return df


def evaluate_model(y_true, y_pred, mean: bool = False, model: str = ''):
    """Evalaute predictions  and return scores

    Args:
        y_true (np.array or list): true labels, either list or np.array
        y_pred (np.array or list): predicted labels, either list or np.array
        mean (bool, optional): if multiple predictions are given (e.g. CV ). Defaults to False.
        model (str, optional): mame of the model. Defaults to ''.

    Returns:
        tuple[float, float, float, float]: MCC, F1, balanced accuracy, ROC-AUC
    """
    if mean:
        mcc, f1, acc, roc_auc = [], [], [], []
        for y in y_pred:
            mcc.append(matthews_corrcoef(y_true, y))
            f1.append(f1_score(y_true, y))
            acc.append(balanced_accuracy_score(y_true, y))
            roc_auc.append(roc_auc_score(y_true, y))
        mcc = np.mean(np.array(mcc))
        f1 = np.mean(np.array(f1))
        acc = np.mean(np.array(acc))
        roc_auc = np.mean(np.array(roc_auc))
    else:
        mcc = matthews_corrcoef(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        acc = balanced_accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
    print(model)
    print('------------------------------------------')
    print("MCC:              ", round(mcc, 4))
    print("f1:               ", round(f1, 4))
    print('Accuracy:         ', round(acc, 4))
    print('ROC-AUC:          ', round(roc_auc, 4))
    print('')

    return [mcc, f1, acc, roc_auc]
