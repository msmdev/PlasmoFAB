"""Script to generate box plots of model results (list of csv, one per model)
or grid search (single grid_csv)

Raises:
    Exception: If no CSV is provided
"""

import seaborn
from argparse import ArgumentParser
import pandas as pd


def main(args):
    """_summary_

    Args:
        args (dict): Command line arguments

    Raises:
        Exception: if no CSV given
    """
    colors = {
        'One-hot-LR': 'lightgreen',
        'One-hot-SVM': 'green',
        'Oligo-SVM': 'blue',
        'ProtT5-LR': 'orange',
        'ProtT5-SVM': 'red',
        'ESM1b-LR': 'violet',
        'ESM1b-SVM': 'purple'}
    if len(args.csv) > 0:
        model_box_plots(args, colors)
    elif args.grid_csv is not None:
        grid_box_plots(args, colors)
    else:
        raise Exception('csv or grid_csv has to be provided')


def model_box_plots(args, colors):
    """Box plots for list of models

    Args:
        args (dict): Command line arguments
        colors (dict): colors per model as defined in main
    """
    df_list = []
    names = args.names
    #names = args.name.split(',')
    print(names)
    for f in args.csv:
        df_list.append(pd.read_csv(f))
    data = pd.DataFrame(columns=['score', 'name'])
    scores = []
    na = []
    for df, name in zip(df_list, names):
        sc = df[args.score]
        for s in sc:
            data.loc[len(data.index)] = [s, name]
    print(data)
    
    ax1 = seaborn.boxplot(data=data, x='name', y='score', saturation=0.3, palette=colors, hue='name', dodge=False)
    ax1.set_title('Cross-validation MCC')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30)
    ax1.legend()
    ax1.grid(True)
    score_str = 'Cross-validation MCC'
    ax1.set(xlabel='Model', ylabel=score_str)
    fig = ax1.get_figure()
    fig.savefig(args.out, bbox_inches='tight')


def large_box_plots(args, colors):
    """Box plots for list of models

    Args:
        args (dict): Command line arguments
        colors (dict): colors per model as defined in main
    """
    df_list = []
    names = args.names
    for f in args.csv:
        df_list.append(pd.read_csv(f))
    data = pd.DataFrame(columns=['score', 'name', 'test'])
    for df, name in zip(df_list, names):
        te = df['test_mcc'].values
        tr = df['train_mcc'].values
        for train, test in zip(tr, te):
            data.loc[len(data.index)] = [train, name, False]
            data.loc[len(data.index)] = [test, name, True]
    print(data)
    
    ax1 = seaborn.boxplot(data=data, x='name', y='score', saturation=0.3, palette=colors, hue='test', dodge=False)
    ax1.set_title('Cross-validation MCC')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30)
    ax1.legend()
    ax1.grid(True)
    score_str = 'Cross-validation MCC'
    ax1.set(xlabel='Model', ylabel=score_str)
    fig = ax1.get_figure()
    fig.savefig(args.out, bbox_inches='tight')


def grid_box_plots(args, colors):
    """Box plot of grid search

    Args:
        args (dict): CMD args
    """
    df = pd.read_csv(args.grid_csv)
    columns = [f"split{i}_{args.score}" for i in range(0, 50)]
    data_list = []
    c_s = []
    for i in range(len(df.index)):
        c_s.append(df.iloc[i]['param_C'])
        values = df.iloc[i][columns].values.flatten()
        data_list.append(values)
    data = pd.DataFrame(columns=c_s)
    for c, d in zip(c_s, data_list):
        data[c] = d
    names = [f"{args.names} C = {c}" for c in c_s]
    box = seaborn.boxplot(data=data, saturation=0.4)
    box.grid(True)
    box.set(xlabel='C', ylabel='Cross-validation MCC')
    fig = box.get_figure()
    fig.savefig(args.out, bbox_inches="tight")


if __name__ == '__main__':
    ###
    ### python3 scripts/plot_results.py --out test.png --score test_mcc --csv [csv-list] --names [name-list]
    ### python3 scripts/plot_results.py --out test.png --grid_csv file.csv --names model_name
    ###
    parser = ArgumentParser('Parse arguments')
    parser.add_argument('--out', help='File to write to')
    parser.add_argument('--score', choices=['test_mcc', 'train_mcc', 'test_f1', 'train_t1', 'test_acc', 'train_acc'], required=True, help='Which score to print')
    parser.add_argument('--names', '--names-list', nargs='+', default=[], help='Names of models to print on png, in order of csv files')
    parser.add_argument('--csv', '--csv-list', nargs='+', default=[], help='CSV files to read, one for each model')
    parser.add_argument('--name', type=str, help='name of single model')
    parser.add_argument('--grid_csv', default=None, help='Single CSV file for plotting grid search results')
    parser.add_argument('--train', help='S')
    args = parser.parse_args()
    main(args)
