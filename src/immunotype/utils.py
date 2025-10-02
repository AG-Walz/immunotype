import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from .constants import TOKEN_VOCABULARY
import itertools


def tokenize(sequences, max_len):
    return (
        np.array([
            [TOKEN_VOCABULARY[e] for e in seq.split()] +
            [0 for _ in range(max_len - len(seq.split()))]
            for seq in sequences])
    )


def get_hetero_data(peptide_df, mhcs, max_n_peptides):
    """
    Get HeteroData object from HLA and peptide features.
    :param mhc_features:
    :param peptide_features:
    :return:
    """
    mhc_mhc_edges = torch.concat([
        torch.arange(len(mhcs)).unsqueeze(0),
        torch.arange(len(mhcs)).unsqueeze(0)
    ], 0)  # MHCs only connected to themselves
    peptide_features = ['[CLS] ' + ' '.join(list(peptide)) + ' [SEP]' for peptide in peptide_df['peptide'].values]
    peptide_features = tokenize(peptide_features, pd.Series(peptide_features).str.split().str.len().max())

    data = []
    for sample, idx in peptide_df.groupby('sample').groups.items():
        for i in range(0, len(idx), max_n_peptides):
            entry = HeteroData()
            entry['peptide'].x = torch.tensor(peptide_features[idx[i:i+max_n_peptides]], dtype=torch.int32)
            entry['mhc'].x = torch.tensor(mhcs, dtype=torch.int32)
            edges = torch.tensor(
                list(itertools.product(range(len(entry['peptide'].x)), range(len(entry['mhc'].x)))),
                dtype=torch.int32
            )
            entry['peptide', 'determines', 'mhc'].edge_index = edges.T
            entry['peptide', 'influences', 'peptide'].edge_index = torch.concat([
                torch.arange(len(entry['peptide'].x)).unsqueeze(0),
                torch.arange(len(entry['peptide'].x)).unsqueeze(0)
            ], 0)
            entry['mhc', 'influences', 'mhc'].edge_index = mhc_mhc_edges
            entry.sample = sample
            data.append(entry)
    return data


def load_weights(model, gnn_weight_path):
    state_dict = torch.load(gnn_weight_path, weights_only=True, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=True)
    return model


def create_typing_summary(typing_df):
    """
    Create a typing summary DataFrame in the format: sample, alleles
    with semicolon-separated alleles per sample.
    
    Args:
        typing_df: DataFrame with columns ['sample', 'locus', 'allele']
        
    Returns:
        DataFrame with columns ['sample', 'alleles']
    """
    if len(typing_df) == 0:
        return pd.DataFrame(columns=['sample', 'alleles'])
    
    summary = typing_df.groupby('sample')['allele'].apply(
        lambda x: ';'.join(sorted(x.values))
    ).reset_index()
    summary.columns = ['sample', 'alleles']
    return summary
