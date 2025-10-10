import sys
import warnings
from pathlib import Path

import pandas as pd
import rich_click as click

from .constants import ASCII_BANNER
from .constants import __authors__
from .immunotype import predict
from .utils import create_typing_summary

# Import version from package
try:
    from . import __version__
except ImportError:
    __version__ = "unknown"

# Configure rich-click for better looking CLI
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.STYLE_ERRORS_SUGGESTION = "magenta italic"
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True

# Get package root directory
PACKAGE_ROOT = Path(__file__).parent


def show_banner():
    """Display the ASCII banner with authors if not showing version."""
    # Check if --version is in the command line arguments
    if "--version" not in sys.argv:
        click.secho(ASCII_BANNER, fg="cyan", bold=True)
        authors_text = f"Authors: {', '.join(__authors__)}"
        click.secho(authors_text, fg="bright_blue", dim=True)
        click.echo()  # Add a blank line after banner


@click.command(
    help="Predict HLA typing from immunopeptide sequences.\n\n"
    "This tool uses graph neural networks and lookup tables to predict HLA allele typing "
    "from immunopeptidomics data. Provide a peptide input file and optionally customize "
    "the HLA alleles to consider.",
    epilog="For more information, visit: https://github.com/immunotype/immunotype",
)
@click.argument(
    "input",
    type=click.Path(exists=True, path_type=Path),
    help="TSV input file. Either a single column of peptides or two columns with sample IDs and peptides.",
)
@click.argument(
    "output",
    type=click.Path(path_type=Path),
    help="Path to save the typing output (format depends on input type).",
)
@click.option(
    "--hla-input",
    type=click.Path(path_type=Path),
    default=PACKAGE_ROOT / "data" / "selected_alleles.csv",
    help="Path to the HLA input file containing alleles to consider.",
    show_default=True,
)
@click.option(
    "--out_probs",
    type=click.Path(path_type=Path),
    help="Save detailed HLA probabilities to specified TSV file.",
)
@click.option(
    "--batch-size",
    default=1,
    type=int,
    help="Number of samples to predict simultaneously.",
    show_default=True,
)
@click.option(
    "--max-n-peptides",
    default=10_000,
    type=int,
    help="Maximum number of peptides to predict at once.",
    show_default=True,
)
@click.option("--no-gnn", is_flag=True, help="Disable the pre-trained GNN model.")
@click.option("--no-lookup", is_flag=True, help="Disable the lookup table.")
@click.option(
    "--gnn-weight-path",
    type=click.Path(path_type=Path),
    default=PACKAGE_ROOT / "weights" / "gnn_model_weights.pt",
    help="Path to GNN model weights file.",
    show_default=True,
)
@click.version_option(version=__version__, prog_name="immunotype")
def main(
    input: Path,
    hla_input: Path,
    output: Path,
    out_probs: Path,
    batch_size: int,
    max_n_peptides: int,
    no_gnn: bool,
    no_lookup: bool,
    gnn_weight_path: Path,
):
    """
    Predict HLA typing from peptide sequences using immunotype.

    PEPTIDE_FILE should contain peptide sequences, either as a single column
    or with headers including 'peptide' and optionally 'sample' columns.
    """

    # Show the ASCII banner
    show_banner()

    # Validate that at least one method is enabled
    if no_gnn and no_lookup:
        click.secho(
            "❌ Error: Cannot disable both GNN and lookup methods!", fg="red", err=True
        )
        raise click.Abort()

    # Load and process peptide data
    input_df = pd.read_csv(input, sep="\t", header=None)

    # Determine input type based on structure
    if input_df.shape[1] == 1:
        # Single column peptide list (like peptide_list.tsv)
        peptide_df = input_df.copy()
        peptide_df.columns = ["peptide"]
        peptide_df["sample"] = 0  # All peptides belong to sample 0
        is_multi_sample = False
        click.secho(
            f"✅ Loaded {len(peptide_df)} peptides (single sample) from {input}",
            fg="green",
        )

    elif input_df.shape[1] == 2:
        # Two column format (like dataset.tsv): sample_id, peptide
        peptide_df = input_df.copy()
        if str(input_df.iloc[0, 0]).lower() in ["sample", "sample_id", "id"]:
            # Has header, skip first row
            peptide_df = pd.DataFrame(
                input_df.iloc[1:].values, columns=["sample", "peptide"]
            )
        else:
            # No header, assign column names
            peptide_df.columns = ["sample", "peptide"]
        is_multi_sample = True
        click.secho(
            f"✅ Loaded {len(peptide_df)} peptides from {len(peptide_df['sample'].unique())} samples from {input}",
            fg="green",
        )
    else:
        raise ValueError(
            f"Unsupported input format. Expected 1 or 2 columns, got {input_df.shape[1]}"
        )

    # Make sure columns are correct types
    peptide_df["sample"] = peptide_df["sample"].astype(str)

    # Load HLA alleles
    try:
        selected_alleles = pd.read_csv(hla_input, header=None)[0].values
        click.secho(
            f"✅ Loaded {len(selected_alleles)} HLA alleles from {hla_input}",
            fg="green",
        )
    except Exception as e:
        click.secho(f"❌ Error loading HLA file: {e}", fg="red", err=True)
        raise click.Abort()

    # Make predictions
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pred_df, typing = predict(
            peptide_df=peptide_df,
            selected_alleles=selected_alleles,
            use_gnn=not no_gnn,
            use_lookup=not no_lookup,
            batch_size=batch_size,
            max_n_peptides=max_n_peptides,
            gnn_weight_path=gnn_weight_path,
        )

        # Show any warnings
        click.secho("\n")
        for warning in w:
            click.secho(f"⚠️  {warning.message}", fg="yellow")
        click.secho("\n")

    if is_multi_sample:
        typing_summary = create_typing_summary(typing)
        typing_summary.to_csv(output, sep="\t", index=False)
        click.secho(f"💾 Typing summary saved to {output}", fg="green")
    else:
        alleles = ";".join(sorted(typing["allele"].values))
        with open(output, "w") as f:
            f.write(alleles + "\n")
        click.secho(f"💾 Typing saved to {output}", fg="green")

    if out_probs:
        # Save detailed predictions (like example_prediction.tsv)
        pred_df.to_csv(out_probs, sep="\t", index=False)
        click.secho(f"💾 Probability details saved to {out_probs}", fg="green")

    # Display results
    click.secho("\n🎯 Predicted HLA typing", fg="green")
    for sample, group in typing.groupby("sample"):
        alleles = ", ".join(sorted(group["allele"].values))
        click.secho(f"   Sample {sample}: {alleles}", fg="cyan")


if __name__ == "__main__":
    main()
