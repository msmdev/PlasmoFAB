"""Creates the plasmoFAB dataset from plasmoDB sequences."""
import numpy as np
import pandas as pd
from argparse import ArgumentParser

from pfmptool.utils import plasmodb_fasta_to_dataframe, pfal_dataframe_to_fasta


def _create_dataframe(paths):
    
    # 0. Read sequence FASTA

    plasmodb_df = plasmodb_fasta_to_dataframe(paths['sequences'])
    # drop unnecessary columns
    plasmodb_df.drop(columns=[
        'id','gene_description','pseudo'], inplace=True)
    print(f"Create dataframe from {len(plasmodb_df)} sequences.")

    # 1. Include epitope data
    #       - drop unnecessary columns and rename for merging
    #        - create columns 'epitope_count' and 'confidence'

    epitopes = pd.read_csv(paths['epitopes'])
    epitopes.drop(columns=['Gene ID'], inplace=True)
    epitopes.rename(columns={
        'source_id': 'transcript',
        'Confidence': 'confidence',
        'Epitope Count': 'epitope_count'}, inplace=True)
    plasmodb_df = pd.merge(plasmodb_df, epitopes, on='transcript', how='left')
    plasmodb_df.epitope_count = plasmodb_df.epitope_count.fillna('0').astype('int')
    plasmodb_df.confidence = plasmodb_df.confidence.astype('string')

    # 2. Include evidence data for epitopes

    ep_evid = pd.read_csv(paths['epitopes_evidence'])
    ep_evid.drop(columns=[
        'Gene ID', 'Product',
        'Support for Evidence Code Assignment'], inplace=True)
    ep_evid.rename(columns={
        'Transcript ID(s)': 'transcript',
        'Evidence Code': 'evidence',
        'Reference': 'reference'}, inplace=True)
    # contains only the valid samples with evidence
    ep_evid = ep_evid.loc[ep_evid['evidence'].notna()]


    # 3. include PEXEL and HT evidence

    pexel_evid = pd.read_csv(paths['pexel_evidence'])
    pexel_evid.drop(columns=[
        'Gene ID', 'Product',
        'Support for Evidence Code Assignment'], inplace=True)
    pexel_evid.rename(columns={
        'Transcript ID(s)': 'transcript',
        'Evidence Code': 'evidence',
        'Reference': 'reference'}, inplace=True)
    # only keep samples with evidence provided
    pexel_evid = pexel_evid.loc[pexel_evid['evidence'].notna()]
    plasmodb_df['PEXEL_motif'] = np.where(plasmodb_df['transcript'].isin(
        pexel_evid.transcript.unique()), True, False
    )

    ht_evid = pd.read_csv(paths['ht_evidence'])
    ht_evid.drop(columns=[
        'Gene ID', 'Product',
        'Support for Evidence Code Assignment'], inplace=True)
    ht_evid.rename(columns={
        'Transcript ID(s)': 'transcript',
        'Evidence Code': 'evidence',
        'Reference': 'reference'}, inplace=True)
    ht_evid = ht_evid.loc[ht_evid['evidence'].notna()]
    plasmodb_df['HT_motif'] = np.where(plasmodb_df['transcript'].isin(
        ht_evid.transcript.unique()), True, False
    )

    # 4. include RIFIN & STEVOR evidence

    rif_stev_evid = pd.read_csv(paths['rif_stev_evidence'])
    rif_stev_evid.drop(columns=[
        'Gene ID', 'Product',
        'Support for Evidence Code Assignment'], inplace=True)
    rif_stev_evid.rename(columns={
        'Transcript ID(s)': 'transcript',
        'Evidence Code': 'evidence',
        'Reference': 'reference'}, inplace=True)
    rif_stev_evid = rif_stev_evid.loc[rif_stev_evid['evidence'].notna()]

    # 5. join evidence tables with plasmodb
    #       - adds the columns 'evidence' and 'reference' to the table

    evidence = pd.concat([
        ep_evid,
        pexel_evid,
        ht_evid,
        rif_stev_evid], ignore_index=True, join='outer')
    # some samples are doubled: 'PF...,PF...'. remove after comma
    evidence.transcript = np.where(
        evidence.transcript.str.contains(','),
        evidence.transcript.str.split(',')[0],
        evidence.transcript)
    # only keep samples with reference publication provided
    evidence = evidence.loc[evidence.reference.notna()]
    evidence =  evidence.drop_duplicates(subset='transcript', keep='first')

    plasmodb_df = pd.merge(plasmodb_df, evidence, on='transcript', how='left')
    plasmodb_df['evidence'] = plasmodb_df['evidence'].astype('string')
    plasmodb_df.reference = plasmodb_df.reference.astype('string')

    # 6. include sporozoite surface-proteins

    sporo = pd.read_csv(paths['sporozoites'], sep=';')
    sporo = sporo.rename(columns={
        'Protein': 'gene',
        'Priority Tier (1 is highest)': 'mass_spec_prio',
        'Signal/TM/GPIb': 'signal/TM/GPIb'}
    )
    sporo = sporo.loc[sporo['mass_spec_prio'] <= 3]
    sporo.drop(columns=sporo.columns.difference([
        'gene', 'mass_spec_prio',
        'signal/TM/GPIb']), inplace=True)
    sporo['transcript'] = sporo['gene']
    sporo.transcript = sporo.transcript.astype('string') + '.1'
    # for one gene there exist two transcripts. Naming according to plasmodb_df
    sporo.loc[sporo.gene == 'PF3D7_1411100.2', 'transcript'] = 'PF3D7_1411100.2'
    sporo.loc[sporo.gene == 'PF3D7_1411100.1', 'transcript'] = 'PF3D7_1411100.1'

    plasmodb_df = pd.merge(plasmodb_df, sporo, on='transcript', how='left')
    plasmodb_df['mass_spec_prio'] = plasmodb_df['mass_spec_prio'].fillna(0).astype('int')
    plasmodb_df['signal/TM/GPIb'] = plasmodb_df['signal/TM/GPIb'].astype('string')

    plasmodb_df.drop(columns=['signal/TM/GPIb', 'gene_y', 'gene_x'], inplace=True)

    # 7. include known intracellular proteins for negative set
    #       - slice off suffix and append ".1" , ".2" ..., to match
    #           naming convention of plasmoDB for merging

    intras = pd.read_csv(paths['intracellular'], sep=';')
    intras['transcript'] = intras.ID.astype('string')
    intras.drop_duplicates(subset='transcript', keep='first', inplace=True)
    intras.drop(columns=[
        'ID', 'Name', 'evidence', 'reference_2', 'cluster name', 'Unnamed: 5',],
        inplace=True)

    intras.transcript = intras.transcript.str.slice_replace(start=13, repl='.1')
    plasmodb_df['intracellular'] = np.where(plasmodb_df.transcript.isin(
        intras.transcript.unique()), True, False
    )

    # mark encymes as additional negative samples
    plasmodb_df['is_encyme'] = np.where(plasmodb_df.description.str.contains('ase'), True, False)

    # merge with plasmod and combine refrerences
    plasmodb_df = pd.merge(plasmodb_df, intras, on='transcript', how='left')
    plasmodb_df.reference = plasmodb_df.reference.fillna(plasmodb_df['Reference'].astype('string'))
    plasmodb_df.drop(columns=['Reference'])

    # remove duplicates
    plasmodb_df.drop_duplicates(subset='transcript', keep='first', inplace=True)

    #
    # 8. Last step: include proteins reviewed by UniProt
    #
    uniprot_pos = pd.read_csv(paths['uniprot_pos'], sep=';')
    uniprot_neg = pd.read_csv(paths['uniprot_neg'], sep=';')
    uniprot_no_location = pd.read_csv(paths['uniprot_no_location'], sep=';')

    plasmodb_df['uniprot_pos'] = np.where(
        plasmodb_df.transcript.isin(uniprot_pos.transcript.unique()), True, False)
    plasmodb_df['uniprot_neg'] = np.where(
        plasmodb_df.transcript.isin(uniprot_neg.transcript.unique()), True, False)
    plasmodb_df['uniprot_no_loc'] = np.where(
        plasmodb_df.transcript.isin(uniprot_no_location.transcript.unique()), True, False)

    return plasmodb_df


def _select_dataset_samples(plasmodb_df):

    print("IEDB all:        ", len(plasmodb_df.loc[(plasmodb_df.confidence == 'High') | (plasmodb_df.confidence == 'Medium')]))
    print("IEDB high:       ", len(plasmodb_df.loc[((plasmodb_df.confidence == 'High') & (plasmodb_df.reference.notna()))]))
    print("IEDB med:        ", len(plasmodb_df.loc[((plasmodb_df.confidence == 'Medium') & (plasmodb_df.reference.notna()))]))

    print("PEXEL/HT,        ", len(plasmodb_df.loc[((plasmodb_df.HT_motif) | (plasmodb_df.PEXEL_motif)) & (plasmodb_df.reference.notna())]))
    print("mass spec 1,2,3  ", len(plasmodb_df.loc[(plasmodb_df.mass_spec_prio.isin([1, 2, 3]) & (plasmodb_df.reference.notna()))]))
    print("stevor,          ", len(plasmodb_df.loc[(plasmodb_df.description.str.contains('stevor') & (plasmodb_df.reference.notna()))]))
    print("rifin,           ", len(plasmodb_df.loc[(plasmodb_df.description.str.contains('rifin') & (plasmodb_df.reference.notna()))]))
    print("surface,         ", len(plasmodb_df.loc[(plasmodb_df.description.str.contains('surface') & (plasmodb_df.reference.notna()))]))
    print("membrane,        ", len(plasmodb_df.loc[(plasmodb_df.description.str.contains('membrane') & (plasmodb_df.reference.notna()))]))
    print("circumsporozoite ", len(plasmodb_df.loc[(plasmodb_df.description.str.contains('circumsporozoite') & (
        plasmodb_df.reference.notna()))]))

    print("pfemp1,          ", len(plasmodb_df.loc[(plasmodb_df.description.str.contains('PfEMP1') & (plasmodb_df.reference.notna()))]))
    print("trap,            ", len(plasmodb_df.loc[(plasmodb_df.description.str.contains('TRAP') & (plasmodb_df.reference.notna()))]))
    print("serine repeat,   ", len(plasmodb_df.loc[(plasmodb_df.description.str.contains('serine repeat antigen') & (plasmodb_df.reference.notna()))]))
    print("FIKK,            ", len(plasmodb_df.loc[(plasmodb_df.description.str.contains('FIKK') & (plasmodb_df.reference.notna()))]))
    print("GLURP,           ", len(plasmodb_df.loc[(plasmodb_df.description.str.contains('GLURP') & (plasmodb_df.reference.notna()))]))
    print("GPI-anchor,      ", len(plasmodb_df.loc[(plasmodb_df.description.str.contains('GPI-anchor') & (plasmodb_df.reference.notna()))]))
    print("exported,        ", len(plasmodb_df.loc[(plasmodb_df.description.str.contains('exported') & (plasmodb_df.reference.notna()))]))
    print("CLAG             ", len(plasmodb_df.loc[(plasmodb_df.description.str.contains('CLAG') & (plasmodb_df.reference.notna()))]))
    print("PHIST            ", len(plasmodb_df.loc[(plasmodb_df.description.str.contains('PHIST') & (plasmodb_df.reference.notna()))]))
    print("Reviewed UniProt ", len(plasmodb_df.loc[plasmodb_df.uniprot_pos]))

    # 1. find positive samples by conditions
    positives = plasmodb_df.loc[
        ((plasmodb_df.confidence == 'High') & (plasmodb_df.reference.notna())) |
        ((plasmodb_df.confidence == 'Medium') & (plasmodb_df.reference.notna())) |
        ((plasmodb_df.HT_motif) | (plasmodb_df.PEXEL_motif)) & (plasmodb_df.reference.notna()) |
        (plasmodb_df.mass_spec_prio.isin([1, 2, 3]) & (plasmodb_df.reference.notna())) |
        (plasmodb_df.description.str.contains('surface') & (plasmodb_df.reference.notna())) |
        (plasmodb_df.description.str.contains('rifin') & (plasmodb_df.reference.notna())) |
        (plasmodb_df.description.str.contains('stevor') & (plasmodb_df.reference.notna())) |
        (plasmodb_df.description.str.contains('merozoite surface') & (
            plasmodb_df.reference.notna())) |
        (plasmodb_df.description.str.contains('circumsporozoite') & (
            plasmodb_df.reference.notna())) |
        (plasmodb_df.description.str.contains('PfEMP1')) & (plasmodb_df.reference.notna()) |
        (plasmodb_df.description.str.contains('TRAP') & (plasmodb_df.reference.notna())) |
        (plasmodb_df.description.str.contains('serine repeat antigen') & (
            plasmodb_df.reference.notna())) |
        (plasmodb_df.description.str.contains('FIKK') & (plasmodb_df.reference.notna())) |
        (plasmodb_df.description.str.contains('GLURP') & (plasmodb_df.reference.notna())) |
        (plasmodb_df.description.str.contains('CLAG') & (plasmodb_df.reference.notna())) |
        (plasmodb_df.description.str.contains('PHIST') & (plasmodb_df.reference.notna())) |
        (plasmodb_df.description.str.contains('GPI-anchor') & (plasmodb_df.reference.notna())) |
        (plasmodb_df.description.str.contains('membrane') & (plasmodb_df.reference.notna())) |
        (plasmodb_df.description.str.contains('exported') & (plasmodb_df.reference.notna())) |
        (plasmodb_df.uniprot_pos == True)
        ]

    print("Positive samples: ", len(positives))

    # 2. clean positive set from encymes, duplicates and intracellulars

    encymes = positives.loc[positives.description.str.contains('ase')]
    print('positives contains ', len(encymes), ' encymes')
    # keep encymes known to be involved in cytoadherence
    keep = encymes.loc[
        (encymes.description.str.contains('FIKK') & (encymes.reference.notna())) |
        (encymes.description.str.contains('CLAG') & (encymes.reference.notna())) |
        (encymes.description.str.contains('exported') & (encymes.reference.notna()))
    ]
    print(len(keep), " encymes are kept in positive set")
    drop_encymes = list(set(encymes.index) - set(keep.index))
    positives = positives.drop(index=drop_encymes)

    print(len(positives), " samples in positive set before duplicate removal")
    positives = positives.drop_duplicates(subset=['transcript'], keep='first')
    print(len(positives), ' after duplicate removal')

    false_pos = positives.loc[positives.intracellular]
    print(len(false_pos), ' false positives (intracellulars)')

    positives = positives.drop(index=false_pos.index)
    print(len(positives), ' positive samples after false positive removal')

    plasmodb_df['antigen_candidate'] = np.where(
        plasmodb_df.transcript.isin(positives.transcript.unique()), True, False)

    # 3. create negative set:
    # intracellulars and encymes, which are not in positive set, and provided evidence

    encymes = plasmodb_df.loc[
        plasmodb_df.description.str.contains('ase') &
        (plasmodb_df.antigen_candidate == False) &
        (plasmodb_df.evidence.notna())
    ]

    print("Negative encymes: ", len(encymes))
    print("intracellulars: ", len(plasmodb_df.loc[(plasmodb_df.intracellular == True)]))
    print(f"Negatives reviewed by UniProt: {len(plasmodb_df.loc[plasmodb_df.uniprot_neg == True])}")

    negatives = plasmodb_df.loc[
        ((plasmodb_df.intracellular == True) & (plasmodb_df.antigen_candidate == False)) |
        (plasmodb_df.transcript.isin(encymes.transcript.unique()) & (plasmodb_df.antigen_candidate == False)) |
        (plasmodb_df.uniprot_neg == True)
    ]

    # 5. sanity checks: overlap and duplicates?

    overlap = positives.loc[positives.transcript.isin(negatives.transcript.unique())]
    overlap2 = negatives.loc[negatives.transcript.isin(positives.transcript.unique())]

    print("Pos samples in neg set: ", len(overlap))
    print("Neg samples in pos set: ", len(overlap2))

    print("Duplicates in pos set: ", len(positives) - len(
        positives.drop_duplicates(subset='transcript', keep='first'))
    )
    print("Duplicates in neg set: ", len(negatives) - len(
        negatives.drop_duplicates(subset='transcript', keep='first'))
    )

    return positives, negatives, plasmodb_df


def _final_cleaning(pos, neg, plasmodb_df):

    # drop samples contained in UniProt_no_location
    # as for these no specific location can be identified
    pos_drop = pos.loc[pos.uniprot_no_loc]
    neg_drop = neg.loc[neg.uniprot_no_loc]
    pos = pos.drop(index=pos_drop.index)
    neg = neg.drop(index=neg_drop.index)

    # Also drop potential samples from pos which are contained in UniProt_neg,
    # as this source is more reliable
    pos_drop = pos.loc[pos.uniprot_neg]
    pos = pos.drop(index=pos_drop.index)

    # also remove from the data table
    plasmo_drop = plasmodb_df.loc[plasmodb_df.uniprot_no_loc]
    plasmodb_df = plasmodb_df.drop(index=plasmo_drop.index)
    print(f"{len(pos_drop)} samples dropped from positive set, {len(neg_drop)} from negative set by UniProt no location")

    # PF3D7_1401300.1 is known to be exported, thus remove
    # from negatives (PMID:27795395)
    plasmodb_df.loc[plasmodb_df.transcript == 'PF3D7_1401300.1', 'is_encyme'] = False
    neg = neg.drop(index=neg.loc[neg.transcript == 'PF3D7_1401300.1'].index)

    # PF3D7_0302200.1, PF3D7_0302500.1 and PF3D7_0935800.1 involved in cytoadherence -> remove
    plasmodb_df.loc[plasmodb_df.transcript == 'PF3D7_0302200.1', 'is_encyme'] = False
    plasmodb_df.loc[plasmodb_df.transcript == 'PF3D7_0302500.1', 'is_encyme'] = False
    plasmodb_df.loc[plasmodb_df.transcript == 'PF3D7_0935800.1', 'is_encyme'] = False
    neg = neg.drop(index=neg.loc[neg.transcript == 'PF3D7_0302200.1'].index)
    neg = neg.drop(index=neg.loc[neg.transcript == 'PF3D7_0302500.1'].index)
    neg = neg.drop(index=neg.loc[neg.transcript == 'PF3D7_0935800.1'].index)


    # Finally, there can be samples with different trancsript name but same sequence
    # We only keep one of them (they all have the same reference so it doesn't matter)
    pos = pos.drop_duplicates(subset=['seq'], keep='first')
    neg = neg.drop_duplicates(subset=['seq'], keep='first')

    # Sanity check: no duplicates?
    full = pd.concat([pos, neg])
    samples = full.duplicated(subset=['transcript'], keep=False)
    if True in samples:
        print(f"{sum(samples)} duplicates found")

    return pos, neg, plasmodb_df

def _labeling(pos: pd.DataFrame, neg: pd.DataFrame, clusters: pd.DataFrame):
    """Test labels are assigned in a way that each sample in the test set is below
    30% sequence identity to all other samples in either training or test set.
    """
    # standard cluster params: 
    #   - sequence identity: 0.3
    #   - alignment coverage: 0.3
    #   - clustering mode: slow/sensitive

    # concat pos and neg for easier assigning
    pos['type'] = 1
    neg['type'] = 0
    full = pd.concat([pos, neg])

    # Create a column indicating cluster size and merge with pos and neg
    cluster_sizes_dict = dict(clusters.cluster.value_counts())
    cl_sizes = []
    for row in clusters.itertuples():
        cl_sizes.append(cluster_sizes_dict[getattr(row, "cluster")])
    clusters['cluster_size'] = cl_sizes

    pos = pos.merge(clusters, how='inner')
    neg = neg.merge(clusters, how='inner')

    # Randomly sample 30 pos and 30 neg from the single-samples clusters

    pos_singles = pos.loc[pos.cluster_size == 1]
    neg_singles = neg.loc[neg.cluster_size == 1]

    pos_test = pos_singles.sample(n=30, random_state=87)
    neg_test = neg_singles.sample(n=30, random_state=97)

    pos['test'] = np.where(pos.transcript.isin(pos_test.transcript.unique()), True, False)
    neg['test'] = np.where(neg.transcript.isin(neg_test.transcript.unique()), True, False)

    return pos, neg

def read_mmseqs2(file: str) -> pd.DataFrame:
    """Creates cluster DataFrame from file

    Args:
        fasta (str): fasta

    Returns:
        pd.DataFrame: clusters
    """
    df = pd.DataFrame(columns=['transcript', 'cluster'])
    with open(file, "r") as fh:
            line = fh.readline()
            cluster = -1
            while len(line) != 0:
                if line.startswith('Cluster'):
                    cluster = line.split('#')[1].strip()
                    cluster = int(cluster)
                if line.startswith('>'):
                    sample = line.split('|')[0]
                    sample = sample.lstrip(">").rstrip("\n")
                    df.loc[len(df.index)] = [sample, cluster]
                line = fh.readline()
                    
    return df


def main(args):

    
    # paths = paths = {'ht_evidence': args.ht_evidence, 
    #         'pexel_evidence': args.pexel_evidence, 
    #         'sporozoites': args.sporozoites, 
    #         'sequences': args.seqs, 
    #         'epitopes': args.epitopes,
    #         'rif_stev_evidence': args.rif_stev_evidence, 
    #         'epitopes_evidence': args.ep_evidence,
    #         'intracellular': args.intracellulars,
    #         'clusters': args.mmseqs_clusters}

    paths = {'ht_evidence': '../data/plasmo_fab/plasmoDB_files/plasmodb56_ht_evidence_120922.csv', 
            'pexel_evidence': '../data/plasmo_fab/plasmoDB_files/plasmodb56_pexel_evidence_120922.csv', 
            'sporozoites': '../data/plasmo_fab/plasmoDB_files/swearingen_sporozoites_220122.csv', 
            'sequences': '../data/plasmo_fab/plasmoDB_files/plasmodb56_3d7_proteins_250322.fasta', 
            'epitopes': '../data/plasmo_fab/plasmoDB_files/plasmodb56_epitopes_280322.csv',
            'rif_stev_evidence': '../data/plasmo_fab/plasmoDB_files/plasmodb56_rifin_stevor_evidence_280322.csv', 
            'epitopes_evidence': '../data/plasmo_fab/plasmoDB_files/plasmodb56_epitopes_evidence_280322.csv',
            'intracellular': '../data/plasmo_fab/intracellular_proteins_manuscript.csv',
            'uniprot_pos': '../data/plasmo_fab/uniprot/UniProt_pos.csv',
            'uniprot_neg': '../data/plasmo_fab/uniprot/UniProt_neg.csv',
            'uniprot_no_location': '../data/plasmo_fab/uniprot/UniProt_no_location.csv',
            'clusters': '../data/plasmo_fab/mmseqs2_plasmofab_clusters.out'}    
    
    df = _create_dataframe(paths)

    pos, neg, data_table = _select_dataset_samples(df)

    pos, neg, data_table = _final_cleaning(pos, neg, data_table)

    # Cluster file is needed for test labels
    if paths['clusters'] is not None:
        print("Creating labels using cluster file")
        
        clusters = read_mmseqs2(paths['clusters'])
    
        pos, neg = _labeling(pos, neg, clusters)
    
    # If minimal cols, save only transcript, test label and seq in CSV
    if args.all_columns == True:
        pos.to_csv(f"../data/plasmo_fab/{args.out}_pos.csv", index=False)
        neg.to_csv(f"../data/plasmo_fab/{args.out}_neg.csv", index=False)
    else:
        pos_min = pos[['transcript', 'seq', 'test']]
        neg_min = neg[['transcript', 'seq', 'test']]
        pos_min.to_csv(f"../data/plasmo_fab/{args.out}_pos.csv", index=False)
        neg_min.to_csv(f"../data/plasmo_fab/{args.out}_neg.csv", index=False)
    
    if args.save_plasmodb:
        data_table.to_csv(f"{args.out}_full_table.csv")
    



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-out', help='File name to write resulting FASTAs to', default='plasmoFAB')
    # parser.add_argument('-seqs', help='Path to plasmoDB FASTA file')
    # parser.add_argument('-epitopes', help='Path to epitope CSV file')
    # parser.add_argument('-ep_evidence', help='Path to epitope evidence CSV file')
    # parser.add_argument('-pexel_evidence', help='Path to PEXEL motif evidence CSV file')
    # parser.add_argument('-ht_evidence', help='Path to HT motif evidence CSV file')
    # parser.add_argument('-rif_stev_evidence', help='Path to rifin and stevor evidence CSV file')
    # parser.add_argument('-sporozoites', help='Path to mass-spectrometry sporozoites CSV file')
    # parser.add_argument('-intracellulars', help='Path to intracellular proteins CSV file')
    # parser.add_argument('-mmseqs_clusters', help='Path to file with mmseq2 clusters', default=None)
    # parser.add_argument('-save_splits', help='Whether or not to save CV splits as .npy', default=None)
    # parser.add_argument('-save_np', help='Whether or not to save train and test sets as .npy', default=None)
    parser.add_argument('-save_plasmodb', help='Whether or not to save annotated plasmoDB dataframe to csv', default=True)
    parser.add_argument('-all_columns', help='Whether to include all columns in CSV file', default=False)
    args = parser.parse_args()
    main(args)