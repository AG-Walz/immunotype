import itertools
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from model import GNN
from constants import *
from utils import load_weights, get_hetero_data, tokenize


model = None
mhc_df = pd.read_csv('data/mhc_sequences.csv')
mhc_features = None
lookup_db = None


def predict_lookup(peptide_df, selected_alleles):
    lookup_score_df = pd.merge(lookup_db.loc[lookup_db['allele'].isin(selected_alleles)], peptide_df, how='inner')
    index = pd.MultiIndex.from_tuples(
        [[s, allele[4], allele] for s in peptide_df['sample'].unique() for allele in selected_alleles], 
        names=['sample', 'locus', 'allele']
        )
    lookup_score_df = lookup_score_df[['sample', 'locus', 'allele']].value_counts().reset_index()
    lookup_score_df['probability'] = np.cbrt(lookup_score_df['count'])
    lookup_score_df['probability'] = lookup_score_df.groupby(['sample', 'locus'])['probability'].transform(
        lambda x: x / x.max()
        )
    lookup_score_df = lookup_score_df.set_index(['sample', 'locus', 'allele']).reindex(index).fillna(0).sort_index()

    for sample in lookup_score_df.index.get_level_values(0).unique():
        for p in PLACEHOLDERS:
            if p in selected_alleles:
                locus = p[4]
                scores = lookup_score_df.loc[sample, locus].loc[:, 'count']
                lookup_score_df.loc[sample, locus, p] = (
                    scores.nlargest(2).values[-1] <= scores.max() * LOOKUP_HOMOZYGOUS_THRESHOLDS[locus]
                    ) * 1
    
    lookup_score_df = lookup_score_df.reset_index().drop('count', axis=1)
    return lookup_score_df


def predict_model(peptide_df, selected_alleles, batch_size, max_n_peptides):
    selected_mhc_features = mhc_features[mhc_df['allele'].isin(selected_alleles)]

    data = get_hetero_data(peptide_df, selected_mhc_features, max_n_peptides)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    predictions, samples = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            predictions.append(np.reshape(model(batch).cpu().detach().numpy(), (len(batch.sample), -1)))
            samples.append(batch.sample)

    probabilities = pd.concat([
        pd.DataFrame(np.concatenate(samples), columns=['sample']),
        pd.DataFrame(np.concatenate(predictions), columns=selected_alleles)
    ], axis=1).melt(id_vars='sample', var_name='allele', value_name='probability')
    probabilities = probabilities.groupby(['sample', 'allele'])['probability'].mean().reset_index()

    probabilities['locus'] = probabilities['allele'].str[4]
    return probabilities


def prepare_data(
        use_gnn=True,
        use_lookup=True,
        gnn_weight_path="weights/gnn_model_weights.pth",
):
    if use_gnn:
        global model, mhc_features
        model = GNN()
        model = load_weights(model, gnn_weight_path)

        max_len_mhc = mhc_df['sequence'].str.split().str.len().max()
        mhc_features = tokenize(mhc_df['sequence'].values, max_len_mhc)

    if use_lookup:
        global lookup_db
        lookup_db = pd.read_csv('data/lookup_db.csv')


def predict(
        peptide_df,
        selected_alleles,
        use_gnn=True,
        use_lookup=True,
        batch_size=10,
        max_n_peptides=10_000,
        gnn_weight_path="weights/gnn_model_weights.pth",
):
    if not use_gnn and not use_lookup:
        raise ValueError('Must use GNN or lookup or both')

    # prepare data only when necessary
    if (use_gnn and model is None) or (use_lookup and lookup_db is None):
        prepare_data(use_gnn, use_lookup, gnn_weight_path)

    probabilities = {}

    if use_gnn:
        probabilities['model'] = predict_model(peptide_df, selected_alleles, batch_size, max_n_peptides)

    if use_lookup:
        probabilities['lookup'] = predict_lookup(peptide_df, selected_alleles)

    if use_gnn and use_lookup:
        pred_df = pd.merge(
            probabilities['model'],
            probabilities['lookup'],
            on=['sample', 'allele', 'locus'],
            how='outer',
            suffixes=['_model', '_lookup']
        ).fillna(0)
        pred_df['probability'] = pred_df.apply(
            lambda x: (
                    x['probability_model'] * (w := ENSEMBLE_MODEL_WEIGHTS[x['locus']]) +
                    x['probability_lookup'] * (1 - w)
            ), axis=1
        )
        pred_df = pred_df[['sample', 'locus', 'allele', 'probability_model', 'probability_lookup', 'probability']]
    else:
        pred_df = list(probabilities.values())[0]

    pred_df = pred_df.sort_values(['sample', 'locus', 'allele'])
    
    # typing
    typing = pred_df.loc[pred_df['probability'] > 0]
    typing = typing.groupby(['sample', 'locus']).apply(
        lambda x: x.sort_values(by='probability')['allele'].iloc[-2:],
        include_groups=False
    ).reset_index().drop('level_2', axis=1)

    # filter out homozygous placeholder alleles
    typing = typing.loc[~typing['allele'].str.contains('homozygous')]
    return pred_df, typing

