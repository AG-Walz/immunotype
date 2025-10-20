import warnings

from tqdm import tqdm
from typing import TypeVar, Callable
from tqdm.std import tqdm as TqdmType

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from pathlib import Path

from .constants import (
    PACKAGE_ROOT,
    ENSEMBLE_GNN_WEIGHTS,
    LOOKUP_HOMOZYGOUS_THRESHOLDS,
    LOOKUP_DF,
    PLACEHOLDERS,
)
from .model import GNN
from .utils import get_hetero_data, load_weights

T = TypeVar("T")


def get_typing(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get HLA typing per sample, based on predicted typing probabilities.

    Args:
        pred_df (pd.DataFrame): DataFrame containing predicted probabilities from Ensemble, Lookup or GNN.

    Returns:
        pd.DataFrame: DataFrame with predicted HLA typing (top 2 alleles per locus).
    """
    typing_df = pred_df.loc[pred_df["probability"] > 0]
    typing_df = (
        typing_df.groupby(["sample", "locus"])
        .apply(
            lambda x: x.sort_values(by="probability")["allele"].iloc[-2:],
            include_groups=False,
        )
        .reset_index()
    )
    # filter out homozygous placeholder alleles
    typing_df = typing_df.loc[~typing_df["allele"].str.contains("homozygous")]
    typing_df = (
        typing_df[["sample", "allele"]]
        .groupby("sample")["allele"]
        .apply(lambda x: ";".join(sorted(x)))
    ).reset_index()
    typing_df.columns = ["sample", "typing"]
    return typing_df


def predict_lookup(
    peptide_df: pd.DataFrame,
    allele_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Predict HLA typing probabilities and HLA typing for given peptides and alleles using a lookup.

    Args:
        pred_df (pd.DataFrame): DataFrame with columns ['sample', 'peptide'] used for predicting the typing.
        allele_df (pd.DataFrame): DataFrame with the column ['allele'], used to limit the prediction to the contained alleles.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - pred_df: DataFrame with binding probabilities for each peptide and allele.
            - typing_df: DataFrame with predicted HLA typing (top 2 alleles per locus).
    """

    lookup_score_df = pd.merge(
        LOOKUP_DF.loc[LOOKUP_DF["allele"].isin(allele_df["allele"])],
        peptide_df,
        how="inner",
    )
    index = pd.MultiIndex.from_tuples(
        [
            [s, allele[4], allele]
            for s in peptide_df["sample"].unique()
            for allele in allele_df["allele"].values
        ],
        names=["sample", "locus", "allele"],
    )
    lookup_score_df = (
        lookup_score_df[["sample", "locus", "allele"]].value_counts().reset_index()
    )
    # Apply cube root transformation to counts to improve homozygous detection
    lookup_score_df["probability"] = np.cbrt(lookup_score_df["count"])
    lookup_score_df["probability"] = lookup_score_df.groupby(["sample", "locus"])[
        "probability"
    ].transform(lambda x: x / x.max())
    # Reshape df uniformly
    index = pd.MultiIndex.from_tuples(
        [
            [sample, allele[4], allele]
            for sample in peptide_df["sample"].unique()
            for allele in allele_df["allele"]
        ],
        names=["sample", "locus", "allele"],
    )
    lookup_score_df = (
        lookup_score_df.set_index(["sample", "locus", "allele"])
        .reindex(index)
        .fillna(0)
        .sort_index()
    )

    for s in lookup_score_df.index.get_level_values(0).unique():
        for p in PLACEHOLDERS:
            if p in allele_df["allele"]:
                locus = p[4]
                scores = lookup_score_df.loc[s, locus].loc[:, "count"]
                lookup_score_df.loc[s, locus, p] = (
                    scores.nlargest(2).values[-1]
                    <= scores.max() * LOOKUP_HOMOZYGOUS_THRESHOLDS[locus]
                ) * 1

    lookup_score_df = lookup_score_df.reset_index().drop("count", axis=1)
    if (lookup_score_df["probability"] > 0).sum() >= 3:
        lookup_typing_df = get_typing(lookup_score_df)
    else:
        # Create empty typing DataFrame with correct columns
        lookup_typing_df = pd.DataFrame(columns=["sample", "locus", "allele"])
        warnings.warn(
            f"Missing typing results from Lookup, possibly due to insufficient input data. Use GNN or increase input size.",
            stacklevel=2,
        )
    return lookup_score_df, lookup_typing_df


def predict_model(
    peptide_df: pd.DataFrame,
    allele_df: pd.DataFrame,
    batch_size: int = 1,
    max_n_peptides: int = 50_000,
    gnn_weight_path: str | Path | None = None,
    device: str = "cpu",
    progress: Callable[[T], TqdmType] = tqdm,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Predict HLA typing probabilities and HLA typing for given peptides and alleles using a GNN.

    Args:
        peptide_df (pd.DataFrame): DataFrame containing peptide information.
        allele_df (pd.DataFrame): DataFrame with the columns ['allele', 'sequence'].
            The sequences are being used as model input, only contained alleles are being predicted.
        batch_size (int): Number of samples per batch for model inference. Defaults to 1.
        max_n_peptides (int): Maximum number of peptides to process. Defaults to 50,000.
        gnn_weight_path (str or None, optional): Path to GNN model weights. Defaults loads from package
        device (str): Can be used to run the prediction on GPU. If no cuda device can be found, will default to 'cpu' instead.
            Allowed parameters: 'cuda', 'cpu'. Defaults to 'cuda'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - pred_df: DataFrame with binding probabilities for each peptide and allele.
            - typing_df: DataFrame with predicted HLA typing (top 2 alleles per locus).
    """

    if gnn_weight_path is None:
        gnn_weight_path = PACKAGE_ROOT / "weights" / "gnn_model_weights.pt"

    if device == "cuda":
        if not torch.cuda.is_available():
            warnings.warn(
                f"Device was set to '{device}', but no available '{device}' device has been found, defaulting to 'cpu'.",
                stacklevel=2,
            )
            device = "cpu"
    elif device != "cpu":
        raise ValueError(
            f"device was set to '{device}', but only 'cuda' or 'cpu' are allowed."
        )

    model = GNN().to(device)
    model = load_weights(model, gnn_weight_path, device)

    data = get_hetero_data(peptide_df, allele_df, max_n_peptides)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    predictions, samples = [], []
    model.eval()

    with torch.no_grad():
        for batch in progress(dataloader, desc="🤖 Running HLA typing prediction..."):
            predictions.append(
                np.reshape(
                    model(batch.to(device)).cpu().detach().numpy(),
                    (len(batch.sample), -1),
                )
            )
            samples.append(batch.sample)

    probabilities_df = pd.concat(
        [
            pd.DataFrame(np.concatenate(samples), columns=["sample"]),
            pd.DataFrame(np.concatenate(predictions), columns=allele_df["allele"]),
        ],
        axis=1,
    ).melt(id_vars="sample", var_name="allele", value_name="probability")

    probabilities_df = (
        probabilities_df.groupby(["sample", "allele"])["probability"]
        .mean()
        .reset_index()
    )

    probabilities_df["locus"] = probabilities_df["allele"].str[4]
    typing_df = get_typing(probabilities_df)
    probabilities_df = probabilities_df[["sample", "locus", "allele", "probability"]]
    return probabilities_df, typing_df


def predict_ensemble(
    peptide_df: pd.DataFrame,
    allele_df: pd.DataFrame,
    batch_size: int = 1,
    max_n_peptides: int = 50_000,
    gnn_weight_path: str | None = None,
    device: str = "cpu",
    progress: Callable[[T], TqdmType] = tqdm,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Predict HLA typing probabilities and HLA typing for given peptides and alleles using both, GNN and lookup.

    Args:
        peptide_df (pd.DataFrame): DataFrame containing peptide information.
        allele_df (pd.DataFrame): DataFrame with the columns ['allele', 'sequence'].
            The sequences are being used as model input, only contained alleles are being predicted.
        batch_size (int): Number of samples per batch for model inference. Defaults to 1.
        max_n_peptides (int): Maximum number of peptides to process. Defaults to 50,000.
        gnn_weight_path (str or None, optional): Path to GNN model weights. Defaults loads from package
        device (str): Can be used to run the prediction on GPU. If no cuda device can be found, will default to 'cpu' instead.
            Allowed parameters: 'cuda', 'cpu'. Defaults to 'cpu'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - pred_df: DataFrame with binding probabilities for each peptide and allele.
            - typing_df: DataFrame with predicted HLA typing (top 2 alleles per locus).
    """

    # Get predictions
    pred_model_df, _ = predict_model(
        peptide_df,
        allele_df,
        batch_size,
        max_n_peptides,
        gnn_weight_path,
        device,
        progress,
    )
    pred_lookup_df, _ = predict_lookup(peptide_df, allele_df)

    pred_df = pd.merge(
        pred_model_df,
        pred_lookup_df,
        on=["sample", "allele", "locus"],
        how="outer",
        suffixes=["_gnn", "_lookup"],
    ).fillna(0)

    # Apply ensemble weighting
    pred_df["probability"] = pred_df.apply(
        lambda x: (
            x["probability_gnn"] * (w := ENSEMBLE_GNN_WEIGHTS[x["locus"]])
            + x["probability_lookup"] * (1 - w)
        ),
        axis=1,
    )

    pred_df = pred_df[
        [
            "sample",
            "locus",
            "allele",
            "probability_gnn",
            "probability_lookup",
            "probability",
        ]
    ]
    pred_df = pred_df.sort_values(["sample", "locus", "allele"])

    typing_df = get_typing(pred_df)
    return pred_df, typing_df


def predict(
    peptide_df: pd.DataFrame,
    allele_df: pd.DataFrame,
    prediction_model: str = "ensemble",
    batch_size: int = 1,
    max_n_peptides: int = 50_000,
    gnn_weight_path: str | None = None,
    device: str = "cpu",
    progress: Callable[[T], TqdmType] = tqdm,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Predict HLA typing probabilities and HLA typing for given peptides and alleles.

    Args:
        peptide_df (pd.DataFrame): DataFrame containing peptide information.
        allele_df (pd.DataFrame): DataFrame with the columns ['allele', 'sequence'].
            The sequences are being used as model input, only contained alleles are being predicted.
        prediction_model (str): Model
        batch_size (int): Number of samples per batch for model inference. Defaults to 1.
        max_n_peptides (int): Maximum number of peptides to process. Defaults to 50,000.
        gnn_weight_path (str or None, optional): Path to GNN model weights. Defaults loads from package
        device (str): Can be used to run the prediction on GPU. If no cuda device can be found, will default to 'cpu' instead.
            Allowed parameters: 'cuda', 'cpu'. Defaults to 'cpu'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - pred_df: DataFrame with binding probabilities for each peptide and allele.
            - typing_df: DataFrame with predicted HLA typing (top 2 alleles per locus).
    """

    if prediction_model == "ensemble":
        pred_df, typing_df = predict_ensemble(
            peptide_df,
            allele_df,
            batch_size,
            max_n_peptides,
            gnn_weight_path,
            device,
            progress,
        )
    elif prediction_model == "lookup":
        pred_df, typing_df = predict_lookup(peptide_df, allele_df)
    elif prediction_model == "gnn":
        pred_df, typing_df = predict_model(
            peptide_df,
            allele_df,
            batch_size,
            max_n_peptides,
            gnn_weight_path,
            device,
            progress,
        )
    else:
        raise ValueError(
            "prediction_model must be either one of: Ensemble, Lookup or GNN."
        )
    return pred_df, typing_df
