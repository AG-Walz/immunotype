"""
Gradio app interface for immunotype.

This module provides the app-based user interface for HLA typing predictions.
It creates an interactive Gradio interface that allows users to input peptide
sequences and get HLA typing predictions with visualization.

Usage:
    python app.py  # Launch directly
"""

import warnings
import html

from pathlib import Path

import gradio as gr
import pandas as pd

from immunotype.utils import parse_peptide_input, parse_allele_input
from immunotype.immunotype import predict
from immunotype.constants import PREDICTION_MODELS

# Get package root directory
PACKAGE_ROOT = Path(__file__).parent

# number of decimals shown and in export
DECIMAL_PRECISION = 4

typing_df = None
probability_df = None


def submit(
    peptides: str,
    alleles: str,
    max_n_peptides: int,
    batch_size: int,
    prediction_model: str,
    use_gpu: bool,
):
    """Executes the script by pressing the submit button."""
    global typing_df, probability_df

    try:
        peptide_df = parse_peptide_input(peptides)
        allele_df = parse_allele_input(alleles)

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings(
                "ignore",
                message="The PyTorch API of nested tensors is in prototype stage and will change in the near future. "
                + "We recommend specifying layout=torch.jagged when constructing a nested tensor, "
                + "as this layout receives active development, has better operator coverage, and works with torch.compile.",
            )
            probability_df, typing_df = predict(
                peptide_df,
                allele_df,
                prediction_model=prediction_model.lower(),
                max_n_peptides=max_n_peptides,
                batch_size=batch_size,
                device="cuda" if use_gpu else "cpu",
                progress=gr.Progress().tqdm,
            )
            for warning in w:
                gr.Warning(html.escape(str(warning.message)), duration=None)
    except Exception as e:
        raise gr.Error(html.escape(str(e)), duration=None)

    typing_path = "typing.tsv"
    typing_df.to_csv(typing_path, index=False, sep="\t")

    probabilities_path = "probabilities.tsv"
    probability_df.to_csv(
        probabilities_path,
        index=False,
        float_format=f"%.{DECIMAL_PRECISION}f",
        sep="\t",
    )
    return (
        typing_df,
        update_probability_output(),
        gr.update(value=typing_path, visible=True),
        gr.update(value=probabilities_path, visible=True),
    )


def update_probability_output():
    """Format the probability output prediction."""
    global probability_df
    style = probability_df.style.format(
        precision=DECIMAL_PRECISION
    ).background_gradient(cmap="Blues")
    return style


def sort_table(col: str):
    """Sort the table by col."""
    global probability_df
    probability_df = probability_df.sort_values(
        by=col, ascending=(col != "probability")
    )
    return update_probability_output()


def update_peptide_input(file: str):
    """Update the peptide input shown in the text field."""
    peptides = "\n".join(pd.read_csv(file, header=None).iloc[:, 0].values)
    return gr.update(value=peptides)


def update_allele_input(file: str):
    """Update the allele input shown in the text field."""
    alleles = "\n".join(pd.read_csv(file, header=None).iloc[:, 0].values)
    return gr.update(value=alleles)


example_peptides = "\n".join(
    pd.read_csv(
        PACKAGE_ROOT / "examples" / "single_sample_input.tsv",
        header=None,
        sep="\t",
    )[0].values
)

example_alleles = "\n".join(
    pd.read_csv(PACKAGE_ROOT / "data" / "selected_alleles.csv", header=None)[0].values
)


def create_interface():
    """Create the Gradio interface."""
    with gr.Blocks(title="immunotype", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🧬 immunotype")
        gr.Markdown("Peptide-based HLA typing from immunopeptidomics data")

        with gr.Tab("Prediction"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Group():
                        peptide_input = gr.Textbox(
                            label="Peptides input",
                            info="Peptides need to be separated by newlines (example).",
                            lines=26,
                            value=example_peptides,
                        )
                        peptide_file_input = gr.File(label="Peptides input", height=140)
                        _ = gr.ClearButton([peptide_input, peptide_file_input])
                    with gr.Accordion("Additional settings", open=False):
                        with gr.Row():
                            model_toggle = gr.Radio(
                                choices=PREDICTION_MODELS,
                                value="Ensemble",
                                label="Select which model to use",
                                info="Ensemble uses both, the pre-trained graph neural network and the peptide-HLA lookup table. "
                                + "Alternatively, you can also use either of them alone.",
                            )
                        with gr.Group():
                            allele_input = gr.Textbox(
                                label="HLA allele input",
                                info="Alleles need to be separated by newlines."
                                + "Important: changing which alleles to predict is not recommended, see the help tab for further information.",
                                lines=20,
                                value=example_alleles,
                            )
                            allele_file_input = gr.File(
                                label="HLA allele input", height=140
                            )
                            _ = gr.ClearButton([allele_input, allele_file_input])
                        with gr.Group():
                            n_peptides_slider = gr.Slider(
                                1_000,
                                100_000,
                                value=50_000,
                                step=1_000,
                                interactive=True,
                                label="Maximum number of peptides",
                                info="Controls the maximum number of peptides per prediction run. "
                                + "Note that all peptides are predicted and allele probabilities averaged, "
                                + "if the number of peptides is larger than the batch size",
                            )
                            batch_size_slider = gr.Slider(
                                1,
                                100,
                                value=1,
                                step=1,
                                interactive=True,
                                label="Batch size",
                                info="Controls how many samples should be predicted simultaneously. "
                                + "Affects only Ensemble and GNN prediction from the model selection.",
                            )
                            use_gpu = gr.Checkbox(
                                label="Use GPU", info="Predict on GPU instead of CPU."
                            )

                    submit_button = gr.Button("Submit", variant="primary")

                with gr.Column(scale=4):
                    with gr.Group():
                        typing = gr.Dataframe(
                            headers=["sample", "typing"],
                            datatype=["str", "str"],
                            row_count=1,
                            col_count=(2, "fixed"),
                            show_copy_button=True,
                            label="Typing",
                        )
                        typing_output = gr.File(label="Download CSV", visible=False)
                    with gr.Group():
                        col_selector = gr.Dropdown(
                            choices=["sample", "allele", "probability"], label="Sort by"
                        )
                        typing_probabilities = gr.Dataframe(
                            headers=["sample", "allele", "probability", "locus"],
                            datatype=["str", "str", "number", "str"],
                            row_count=1,
                            label="Typing probabilities",
                            col_count=(4, "fixed"),
                            show_copy_button=True,
                        )
                        probability_output = gr.File(
                            label="Download CSV", visible=False
                        )

            _ = submit_button.click(
                submit,
                inputs=[
                    peptide_input,
                    allele_input,
                    n_peptides_slider,
                    batch_size_slider,
                    model_toggle,
                    use_gpu,
                ],
                outputs=[
                    typing,
                    typing_probabilities,
                    typing_output,
                    probability_output,
                ],
            )
            peptide_file_input.upload(
                update_peptide_input, inputs=peptide_file_input, outputs=peptide_input
            )
            allele_file_input.upload(
                update_allele_input, inputs=allele_file_input, outputs=allele_input
            )
            col_selector.change(
                sort_table, inputs=col_selector, outputs=typing_probabilities
            )
        with gr.Tab("Help"):
            gr.Markdown("""
            ## 📚 Tutorial and Resources

            ### Usage
            1. **Input peptides**: Enter peptide sequences separated by newlines, or upload a file
            2. **Select typing method**: Choose between automatic typing or input known HLA alleles
            4. **Submit**: Click submit to run the prediction
            5. **View results**: See predicted typing and download detailed probabilities

            ### Input Formats
            - **Peptides**: One peptide sequence per line (e.g., `ALDGRETD`)
            - **HLA alleles**: One allele per line (e.g., HLA-A*24-27)
            - **Files**: TSV/CSV files with peptide sequences in the first column

            ### Output
            - **Typing**: Predicted HLA alleles for your sample
            - **Probabilities**: Detailed probability scores for all tested alleles
            - **CSV Download**: Full results table for further analysis

            ### Citation
            If you use immunotype in your research, please cite TODO.
            """)

    return app


# Main function to launch the app via CLI
def main():
    app = create_interface()
    app.launch()


# For direct execution of the app interface
if __name__ == "__main__":
    main()
