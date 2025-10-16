import sys
import warnings
from pathlib import Path

import pandas as pd
import rich_click as click

from .constants import __authors__,ASCII_BANNER, PREDICTION_MODELS
from .immunotype import predict
from .utils import parse_peptide_input, parse_allele_input

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

DECIMAL_PRECISION = 4


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
    "peptide_input",
    type=click.Path(exists=True, path_type=Path),
    help="TSV input file. Either a single column of peptides or two columns with sample IDs and peptides.",
)
@click.argument(
    "typing_output",
    type=click.Path(path_type=Path),
    help="Path to save the typing output.",
)
@click.option(
    "--prob_output",
    type=click.Path(path_type=Path),
    help="Save detailed HLA probabilities to specified TSV file.",
)
@click.option(
    "--hla-input",
    type=click.Path(path_type=Path),
    default=PACKAGE_ROOT / "data" / "selected_alleles.csv",
    help="Path to the HLA input file containing alleles to consider.",
    show_default=True,
)
@click.option(
    "--max-n-peptides",
    default=50_000,
    type=int,
    help="Maximum number of peptides to predict at once.",
    show_default=True,
)
@click.option(
    "--prediction_model",
    default="ensemble",
    type=click.Choice([model.lower() for model in PREDICTION_MODELS], case_sensitive=True),
    help="Disable the pre-trained GNN model.",
    show_default=True,
)
@click.version_option(version=__version__, prog_name="immunotype")
def main(
    peptide_input: Path,
    hla_input: Path,
    typing_output: Path,
    prob_output: Path,
    max_n_peptides: int,
    prediction_model: str,
):
    """
    Predict HLA typing from peptide sequences using immunotype.

    PEPTIDE_FILE should contain peptide sequences, either as a single column
    or with headers including 'peptide' and optionally 'sample' columns.
    """

    # Show the ASCII banner
    show_banner()

    # Load and process peptide data
    with open(peptide_input, "r") as file:
        peptide_df = parse_peptide_input(file.read())

    click.secho(
        f"✅ Loaded {len(peptide_df)} peptides from {len(peptide_df['sample'].unique())} samples from {peptide_input}",
        fg="green",
    )

    # Load HLA alleles
    with open(hla_input, "r") as file:
        allele_df = parse_allele_input(file.read())

    click.secho(
        f"✅ Loaded {len(allele_df)} HLA alleles from {hla_input}",
        fg="green",
    )

    # Make predictions
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pred_df, typing_df = predict(
            peptide_df=peptide_df,
            allele_df=allele_df,
            prediction_model=prediction_model,
            max_n_peptides=max_n_peptides,
        )

        # Show any warnings
        click.secho("\n")
        for warning in w:
            click.secho(f"⚠️  {warning.message}", fg="yellow")
        click.secho("\n")

    typing_df.to_csv(typing_output, sep="\t", index=False)
    click.secho(f"💾 Typing saved to {typing_output}", fg="green")

    if prob_output:
        # Save detailed predictions (like example_prediction.tsv)
        pred_df.to_csv(
            prob_output, sep="\t", index=False, float_format=f"%.{DECIMAL_PRECISION}f"
        )
        click.secho(f"💾 Probability details saved to {prob_output}", fg="green")

    # Display results
    click.secho("\n🎯 Predicted HLA typing", fg="green")
    for s, group in typing_df.groupby("sample"):
        alleles = ", ".join(sorted(group["allele"].values))
        click.secho(f"   Sample {s}: {alleles}", fg="cyan")


if __name__ == "__main__":
    main()
