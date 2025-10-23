import itertools

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import Series
import re
from io import StringIO
from pathlib import Path

import torch
from torch.nn import Module
from torch_geometric.data import HeteroData

from .constants import AMINO_ACIDS, TOKEN_VOCABULARY, MHC_SEQUENCE_DF

VALID_AA = re.compile(rf"^[{''.join(AMINO_ACIDS)}]+$")


class PeptideInputSchema(pa.DataFrameModel):
    """Validation schema for peptide input."""

    sample: Series[str] = pa.Field(nullable=False)
    peptide: Series[str] = pa.Field(nullable=False)

    class Config:
        coerce = True
        strict = False  # ignore additional columns

    @pa.check("peptide", name="valid_peptide")
    def peptide_check(cls, peptide: Series[str]) -> Series[bool]:
        return peptide.str.match(VALID_AA)


class AlleleInputSchema(pa.DataFrameModel):
    """Validation schema for allele input."""

    allele: Series[str] = pa.Field(nullable=False)

    class Config:
        coerce = True
        strict = True

    @pa.check("allele", name="valid_allele")
    def allele_check(cls, allele: Series[str]) -> Series[bool]:
        return allele.isin(MHC_SEQUENCE_DF["allele"])


def parse_peptide_input(data: str | pd.DataFrame) -> pd.DataFrame:
    """Parse peptide data from multiple possible formats into a validated DataFrameModel."""
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.read_csv(StringIO(data), sep=r"[,\t; ]+", engine="python")
        if df.shape[1] == 1:
            df.columns = ["peptide"]
            df["sample"] = "sample_0"  # add placeholder sample
        elif {"sample", "peptide"}.issubset(df.columns):
            df = df
        else:
            raise ValueError(
                "Input must have either one column (peptide) or two columns (sample, peptide)."
            )
    df = PeptideInputSchema.validate(df)
    return df


def parse_allele_input(data: str | pd.DataFrame) -> pd.DataFrame:
    """Parse allele data from multiple possible formats into a validated DataFrameModel."""
    if isinstance(data, pd.DataFrame):
        df = data

    else:
        df = pd.read_csv(StringIO(data), sep=r"[,\t; ]+", engine="python")
        if df.shape[1] == 1:
            df.columns = ["allele"]
        elif {"allele"}.issubset(df.columns):
            df = df[["allele"]]
        else:
            raise ValueError("Input must have one column (allele).")
    df = AlleleInputSchema.validate(df)
    return df


def tokenize(sequences: pd.Series) -> npt.ArrayLike:
    """
    Tokenize sequences into integer arrays based on TOKEN_VOCABULARY.
    """
    splits = sequences.str.split()
    max_len = splits.str.len().max()
    return np.array(
        [
            [TOKEN_VOCABULARY[e] for e in seq] + [0 for _ in range(max_len - len(seq))]
            for seq in splits.values
        ]
    )


def get_hetero_data(
    peptide_df: pd.DataFrame, mhc_df: pd.DataFrame, max_n_peptides: int
) -> list[HeteroData]:
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
            Peptide sequences have to be [CLS] and [SEP] enclosed and the amino acids separated by whitespace.
        mhc_df (pd.DataFrame): DataFrame containing [CLS] and [SEP] enclosed, whitespace separated amino acid sequences of alleles.
        max_n_peptides (int): Maximum number of peptides per sample chunk.

    Returns:
        List[HeteroData]: List of HeteroData objects, one per sample chunk.
    """

    peptide_features = tokenize(peptide_df["sequence"])
    mhc_features = tokenize(mhc_df["sequence"])

    # MHCs only connected to themselves
    mhc_mhc_edges = torch.tensor(
        np.repeat(np.expand_dims(np.arange(len(mhc_features)), 0), 2, 0),
        dtype=torch.int32,
    )

    data = []
    for sample, idx in peptide_df.reset_index().groupby("sample").groups.items():
        for i in range(0, len(idx), max_n_peptides):
            entry = HeteroData()
            entry["peptide"].x = torch.tensor(
                peptide_features[idx[i : i + max_n_peptides]], dtype=torch.int32
            )
            entry["mhc"].x = torch.tensor(mhc_features, dtype=torch.int32)
            # Fully connect peptides to MHCs
            edges = torch.tensor(
                list(
                    itertools.product(
                        range(len(entry["peptide"].x)), range(len(entry["mhc"].x))
                    )
                ),
                dtype=torch.int32,
            )

            entry["peptide", "determines", "mhc"].edge_index = edges.T
            entry["peptide", "influences", "peptide"].edge_index = torch.tensor(
                np.repeat(np.expand_dims(np.arange(len(entry["peptide"].x)), 0), 2, 0),
                dtype=torch.int32,
            )
            entry["mhc", "influences", "mhc"].edge_index = mhc_mhc_edges
            entry.sample = sample
            data.append(entry)
    return data


def load_weights(model: Module, gnn_weight_path: str | Path, device: str) -> Module:
    """Load model weights from file."""
    state_dict = torch.load(
        gnn_weight_path, weights_only=True, map_location=torch.device(device)
    )
    model.load_state_dict(state_dict, strict=True)
    return model
