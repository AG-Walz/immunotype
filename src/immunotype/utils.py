import itertools

import numpy as np
import pandas as pd
import torch
from torch.nn import Module
from torch_geometric.data import HeteroData

from .constants import TOKEN_VOCABULARY


def tokenize(sequences, max_len) -> np.ndarray:
    """Tokenize sequences into integer arrays based on TOKEN_VOCABULARY."""
    return np.array(
        [[TOKEN_VOCABULARY[e] for e in seq.split()] + [0 for _ in range(max_len - len(seq.split()))] for seq in sequences]
    )


def get_hetero_data(peptide_df: pd.DataFrame, mhcs, max_n_peptides):
    """
    Constructs a list of HeteroData objects representing peptide-MHC relationships for each sample.

    Each HeteroData object contains peptide and MHC node features, and edge indices for:
    - Peptide-to-MHC ("determines")
    - Peptide-to-peptide ("influences")
    - MHC-to-MHC ("influences")

    Peptide features are tokenized and padded. Peptides are fully connected to all MHCs.
    Peptides and MHCs are also self-connected.

    Args:
        peptide_df (pd.DataFrame): DataFrame containing peptide sequences and sample identifiers.
        mhcs: Array-like object containing MHC features.
        max_n_peptides (int): Maximum number of peptides per sample chunk.

    Returns:
        List[HeteroData]: List of HeteroData objects, one per sample chunk.
    """
    # MHCs only connected to themselves
    mhc_mhc_edges = torch.concat([torch.arange(len(mhcs)).unsqueeze(0), torch.arange(len(mhcs)).unsqueeze(0)], 0)  
    # Tokenize peptides
    peptide_features = ["[CLS] " + " ".join(list(peptide)) + " [SEP]" for peptide in peptide_df["peptide"].values]
    peptide_features = tokenize(peptide_features, pd.Series(peptide_features).str.split().str.len().max())

    data = []
    for sample, idx in peptide_df.groupby("sample").groups.items():
        for i in range(0, len(idx), max_n_peptides):
            entry = HeteroData()
            entry["peptide"].x = torch.tensor(peptide_features[idx[i : i + max_n_peptides]], dtype=torch.int32)
            entry["mhc"].x = torch.tensor(mhcs, dtype=torch.int32)
            # Fully connect peptides to MHCs
            edges = torch.tensor(list(itertools.product(range(len(entry["peptide"].x)), range(len(entry["mhc"].x)))), dtype=torch.int32)

            entry["peptide", "determines", "mhc"].edge_index = edges.T
            entry["peptide", "influences", "peptide"].edge_index = torch.concat(
                [torch.arange(len(entry["peptide"].x)).unsqueeze(0),
                torch.arange(len(entry["peptide"].x)).unsqueeze(0)],
                0
            )
            entry["mhc", "influences", "mhc"].edge_index = mhc_mhc_edges
            entry.sample = sample
            data.append(entry)
    return data


def load_weights(model: Module, gnn_weight_path) -> Module:
    """Load model weights from file."""
    state_dict = torch.load(gnn_weight_path, weights_only=True, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=True)
    return model


def create_typing_summary(typing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a typing summary DataFrame in the format: sample, alleles
    with semicolon-separated alleles per sample.

    Args:
        typing_df: DataFrame with columns ['sample', 'locus', 'allele']

    Returns:
        DataFrame with columns ['sample', 'alleles']
    """
    if len(typing_df) == 0:
        return pd.DataFrame(columns=["sample", "alleles"])

    summary = (
        typing_df.groupby("sample")["allele"]
        .apply(lambda x: ";".join(sorted(x.values)))
        .reset_index()
    )
    summary.columns = ["sample", "alleles"]
    return summary
