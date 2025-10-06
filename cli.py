import pandas as pd
import click
from immunotype import predict


@click.command()
@click.argument("peptide_input_file", type=click.Path(exists=True))
@click.option("--hla_input_file", type=click.Path(), default="data/selected_alleles.csv", help="Path to the optional HLA input file")
@click.option("--output_file", type=click.Path(), help="Path to save the optional output TSV file.")
@click.option("--batch_size", default=1, help="Number of samples to predict simultaneously.")
@click.option("--max_n_peptides", default=10_000, help="Number of maximum number of peptides to predict at once")
@click.option("--no_gnn", is_flag=True, help="Do not use the pre-trained GNN model.")
@click.option("--no_lookup", is_flag=True, help="Do not use the lookup.")
@click.option("--gnn_weight_path", default="weights/gnn_model_weights.pth", help="GNN model weights path")
def main(
        peptide_input_file,
        hla_input_file,
        output_file,
        batch_size,
        max_n_peptides,
        no_gnn,
        no_lookup,
        gnn_weight_path
):
    """
    Predict HLA typing from peptide sequences.
    :param peptide_input_file:
    :param hla_input_file:
    :param output_file:
    :param batch_size:
    :param max_peptides:
    :param no_gnn:
    :param no_lookup:
    :param gnn_weight_path:
    :return:
    """

    peptide_df = pd.read_csv(peptide_input_file, sep="\t", header=None)
    if 'peptide' in peptide_df.iloc[0].values:  # input is a dataframe containing a peptide column
        peptide_df = pd.DataFrame(peptide_df.iloc[1:].values, columns=peptide_df.iloc[0].values)
    if peptide_df.shape[1] == 1:  # input is a peptide list without header
        peptide_df.columns = ["peptide"]
    if not 'sample' in peptide_df.columns:  # sample is missing, all peptides belong to the same sample
        peptide_df['sample'] = 0
    
    peptide_df['sample'] = peptide_df["sample"].astype(str)
    selected_alleles = pd.read_csv(hla_input_file, header=None)[0].values

    pred_df, typing = predict(
        peptide_df=peptide_df,
        selected_alleles=selected_alleles,
        use_gnn=not no_gnn,
        use_lookup=not no_lookup,
        batch_size=batch_size,
        max_n_peptides=max_n_peptides,
        gnn_weight_path=gnn_weight_path,
    )

    # save predictions
    if output_file:
         pred_df.to_csv(output_file, sep='\t', index=False)

    click.secho('Predicted HLA typing', fg='green', bold=True, underline=True)
    for sample, group in typing.groupby('sample'):
        click.secho(f'Sample {sample}: {", ".join(sorted(group["allele"].values))}')


if __name__ == '__main__':
    main()
