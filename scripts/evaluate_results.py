#!/bin/bash
#
# run like
# python3 scripts/evaluate_results.py --model [DeepLoc | DeepTMHMM] --dataset [pfal | deeploc] --res_file [result_file] --out_file [out_file]
#
from pfmptool.utils import evaluate_model, deeploc_fasta_to_dataframe, deeploc_membrane_fasta_to_dataframe

import pandas as pd
from argparse import ArgumentParser
import numpy as np
import os


def main(model, res_file, dataset, out_file):

    if dataset == 'pfal':
        df_dataset = pd.read_csv('data/pfal/pfal.csv')
    if dataset == 'plasmoFAB':
        df_dataset = pd.read_csv('data/plasmo_fab/plasmoFAB.csv')
    elif dataset == 'deeploc':
        df_dataset = deeploc_fasta_to_dataframe('data/deeploc/deeploc_data.fasta')
    elif dataset == 'deeploc_mem':
        df_dataset = deeploc_membrane_fasta_to_dataframe('data/deeploc/deeploc_data.fasta')
    else:
        raise Exception('Dataset not supported: must be pfal or deeploc')

    if model == 'DeepLoc':
        scores, test_scores = eval_DeepLoc(res_file, df_dataset=df_dataset)
    elif model == 'DeepLoc-membrane':
        scores, test_scores = eval_DeepLoc_membrane(res_file, df_dataset=df_dataset)
    elif model == 'DeepTMHMM':
        scores, test_scores = eval_DeepTMHMM(res_file, df_dataset=df_dataset)
    elif model == 'TMHMM':
        scores, test_scores = eval_TMHMM(res_file, df_dataset=df_dataset)
    elif model == 'Phobius':
        scores, test_scores = eval_Phobius(res_file, df_dataset=df_dataset)
    elif model == 'DeepLoc1.0':
        scores, test_scores = eval_DeepLoc10(res_file, df_dataset=df_dataset)
    elif model == 'DeepLoc1.0-membrane':
        scores, test_scores = eval_DeepLoc10_membrane(res_file, df_dataset=df_dataset)
    else:
        raise Exception("Model not supported")

    if args.out_file is not None:
        out_file = os.path.join(os.getcwd(), f"{args.out_file}")
        try:
            os.remove(os.path.join(os.getcwd(), out_file))
        except OSError:
            pass

        with open(out_file, 'a') as f:
            f.write(f"Model {str(model)}, dataset {str(dataset)}\n")
            f.write('------------------------------------------\n')
            f.write('------------------------------------------\n')
            f.write('Scores\n')
            f.write('------------------------------------------\n')
            f.write(f"MCC:              {round(scores[0], 4)}\n")
            f.write(f"f1:               {round(scores[1], 4)}\n")
            f.write(f"Bal. Acc:         {round(scores[2], 4)}\n")
            f.write('\n\n')
            f.write('Test set scores\n')
            f.write('------------------------------------------\n')
            f.write(f"MCC:              {round(test_scores[0], 4)}\n")
            f.write(f"f1:               {round(test_scores[1], 4)}\n")
            f.write(f"Bal. Acc:         {round(test_scores[2], 4)}\n")


def eval_DeepLoc(res_file, df_dataset):
    res = pd.read_csv(res_file)
    res = res.rename(columns={'Protein_ID': 'transcript'})
    res['prediction'] = np.where(
        res['Localizations'].str.contains('Cell membrane') |
        (res['Localizations'].str.contains('Extracellular')) |
        (res['Signals'].str.contains('Transmembrane')) |
        (res['Signals'].str.contains('Signal peptide')), 1, 0)

    res_merged = pd.merge(res, df_dataset, on='transcript')
    y_true = res_merged.type
    y_pred = res_merged.prediction
    test_y_true = res_merged.loc[res_merged.test == True]['type']
    test_y_pred = res_merged.loc[res_merged.test == True]['prediction']

    scores = evaluate_model(y_pred, y_true, model="DeepLoc 2.0")
    test_scores = evaluate_model(test_y_pred, test_y_true, model='DeepLoc 2.0 on test set')

    return scores, test_scores


def eval_DeepLoc_membrane(res_file, df_dataset):
    res = pd.read_csv(res_file)
    res = res.rename(columns={'Protein_ID': 'transcript'})
    res['prediction'] = np.where(
        res['Localizations'].str.contains('Cell membrane') |
        (res['Signals'].str.contains('Transmembrane')), 1, 0)

    res_merged = pd.merge(res, df_dataset, on='transcript')
    y_true = res_merged.type
    y_pred = res_merged.prediction
    test_y_true = res_merged.loc[res_merged.test == True]['type']
    test_y_pred = res_merged.loc[res_merged.test == True]['prediction']

    scores = evaluate_model(y_pred, y_true, model="DeepLoc 2.0 membrane-only")
    test_scores = evaluate_model(test_y_pred, test_y_true, model='DeepLoc 2.0 on test set')

    return scores, test_scores


def eval_DeepTMHMM(res_file, df_dataset):
    res = parse_DeepTMHMM_result(res_file)
    print(res)

    res_merged = pd.merge(df_dataset, res, on='transcript')
    print(len(res_merged))
    # label 1 (== drug target) if
    # - signal peptide or transmembrane is predited
    # - topology contains M or O (membrane or outside)
    res_merged['y_pred'] = np.where(
        res_merged.pred_type.isin(['SP', 'TM', 'SP+TM']) |
        res_merged.topology.str.contains('M') |
        res_merged.topology.str.contains('O'), 1, 0)

    y_true = res_merged.type
    y_pred = res_merged.y_pred
    test_y_true = res_merged.loc[res_merged.test == True]['type']
    test_y_pred = res_merged.loc[res_merged.test == True]['y_pred']

    scores = evaluate_model(y_true, y_pred, model='DeepTMHMM')
    test_scores = evaluate_model(test_y_true, test_y_pred, model='DeepTMHMM on test set')

    return scores, test_scores


def eval_TMHMM(res_file, df_dataset):
    res = pd.DataFrame(columns=['transcript', 'pred_hel', 'topology'])
    with open(res_file) as fh:
        line = fh.readline()
        while len(line) != 0:
            fields = line.split('\t')
            pred_hel = fields[4].split('=')[1]
            transcript = fields[0].split('|')[0]
            topology = str(fields[5].split('=')[1]).removesuffix('\n')
            res.loc[len(res.index)] = [transcript, pred_hel, topology]
            line = fh.readline()
    res.pred_hel = res.pred_hel.astype('int')
    res['pred_type'] = np.where((res.pred_hel > 0), 1, 0)

    res = pd.merge(res, df_dataset, on='transcript')
    y_true = res.type
    y_pred = res.pred_type
    test_y_true = res.loc[res.test == True]['type']
    test_y_pred = res.loc[res.test == True]['pred_type']

    scores = evaluate_model(y_true, y_pred, model='TMHMM 2.0')
    test_scores = evaluate_model(test_y_true, test_y_pred, model='TMHMM 2.0 on test set')
    return scores, test_scores


def eval_Phobius(res_file, df_dataset):
    res = pd.DataFrame(columns=['transcript', 'tm', 'sp', 'pred_topology'])
    with open(res_file) as fh:
        line = fh.readline() # skip header line
        line = fh.readline()
        while len(line) != 0:
            fields = line.split()
            tm = fields[1]
            sp = fields[2]
            transcript = fields[0]
            pred_topology = fields[3]
            if '|' in transcript:
                transcript = transcript.split('|')[0]
            res.loc[len(res.index)] = [transcript, tm, sp, pred_topology]
            line = fh.readline()
    res.tm = res.tm.astype('int')
    res['pred_type'] = np.where((res.pred_topology.str.contains('o')) | (res.tm > 0), 1, 0)
    print(df_dataset)
    res_merged = pd.merge(res, df_dataset, on='transcript')
    scores = evaluate_model(res_merged.type, res_merged.pred_type, model='Phobius')
    test_scores = evaluate_model(
        res_merged[res_merged.test].type,
        res_merged[res_merged.test].pred_type, model='Phobius')

    return scores, test_scores


def eval_DeepLoc10_membrane(res_file, df_dataset):
    res = pd.read_csv(res_file)
    res = res.rename(columns={'Entry ID': 'transcript'})
    res['prediction'] = np.where(
        (res['Type'].str.contains('Membrane')), 1, 0)
    res_merged = pd.merge(res, df_dataset, on='transcript')
    y_true = res_merged.type
    y_pred = res_merged.prediction
    test_y_true = res_merged.loc[res_merged.test == True]['type']
    test_y_pred = res_merged.loc[res_merged.test == True]['prediction']

    scores = evaluate_model(y_pred, y_true, model="DeepLoc 1.0")
    test_scores = evaluate_model(test_y_pred, test_y_true, model='DeepLoc 1.0 on test set')

    return scores, test_scores


def eval_DeepLoc10(res_file, df_dataset):
    res = pd.read_csv(res_file)
    res = res.rename(columns={'Entry ID': 'transcript'})
    res['prediction'] = np.where(
        res['Localization'].str.contains('Extracellular') |
        (res['Type'].str.contains('Membrane')), 1, 0)
    res_merged = pd.merge(res, df_dataset, on='transcript')
    y_true = res_merged.type
    y_pred = res_merged.prediction
    test_y_true = res_merged.loc[res_merged.test == True]['type']
    test_y_pred = res_merged.loc[res_merged.test == True]['prediction']

    scores = evaluate_model(y_pred, y_true, model="DeepLoc 1.0")
    test_scores = evaluate_model(test_y_pred, test_y_true, model='DeepLoc 1.0 on test set')

    return scores, test_scores


def parse_DeepTMHMM_result(file):
    df = pd.DataFrame(columns=['transcript', 'seq', 'pred_type', 'topology'])
    with open(file) as fh:
        line = fh.readline()
        while len(line) != 0:
            sample = ''
            if line.startswith('>'):
                sample = line.split(' | ')
            else:
                raise Exception('Parse error: line should start with ">"')
            transcript = sample[0].removeprefix('>').strip()
            pred_type = sample[1].removesuffix('\n').strip()
            sequence = fh.readline()
            topology = fh.readline()
            if '|' in transcript:
                transcript = transcript.split('|')[0]
            df.loc[len(df.index)] = [transcript, sequence, pred_type, topology]
            line = fh.readline()
    return df


if __name__ == '__main__':
    parser = ArgumentParser('Parse arguments')
    parser.add_argument('--model', help='Model to evaluate')
    parser.add_argument('--res_file', help='Result file to be read')
    parser.add_argument('--out_file', help='File to write scores to')
    parser.add_argument('--dataset', help='Dataset: pfal or deeploc')
    args = parser.parse_args()
    main(model=args.model, res_file=args.res_file, dataset=args.dataset, out_file=args.out_file)
