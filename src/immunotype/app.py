"""
Gradio app interface for immunotype.

This module provides the app-based user interface for HLA typing predictions.
It creates an interactive Gradio interface that allows users to input peptide
sequences and get HLA typing predictions with visualization.

Usage:
    python app.py  # Launch directly
"""

from pathlib import Path

import gradio as gr
import pandas as pd
import torch

from immunotype.immunotype import predict

# Get package root directory
PACKAGE_ROOT = Path(__file__).parent

DEVICE = torch.device("cpu")

# number of decimals shown and in export
DECIMAL_PRECISION = 4

typing_df = None
probability_df = None


def submit(peptides, alleles, max_n_peptides, model_to_use):
    use_gnn = model_to_use != "Lookup"
    use_lookup = model_to_use != "GNN"
    global typing_df, probability_df

    peptide_df = pd.DataFrame(
        peptides.replace("\n", ",").split(","), columns=["peptide"]
    )
    peptide_df["sample"] = 0
    alleles = pd.Series(alleles.replace("\n", ",").split(","))
    probability_df, typing_df = predict(
        peptide_df,
        alleles,
        use_gnn=use_gnn,
        use_lookup=use_lookup,
        max_n_peptides=max_n_peptides,
        gnn_weight_path=PACKAGE_ROOT / "weights" / "gnn_model_weights.pt",
    )
    typing_df = (
        typing_df[["sample", "allele"]]
        .groupby("sample")["allele"]
        .apply(lambda x: ";".join(x.sort_values(ascending=True)))
    ).reset_index()

    typing_path = "typing.csv"
    typing_df.to_csv(typing_path, index=False)

    probabilities_path = "probabilities.csv"
    probability_df.to_csv(
        probabilities_path, index=False, float_format=f"%.{DECIMAL_PRECISION}f"
    )

    return (
        typing_df,
        update_probability_output(),
        gr.update(value=typing_path, visible=True),
        gr.update(value=probabilities_path, visible=True),
    )


def update_probability_output():
    global probability_df
    style = probability_df.style.format(
        precision=DECIMAL_PRECISION
    ).background_gradient(cmap="Blues")
    return style


def sort_table(col):
    global probability_df
    probability_df = probability_df.sort_values(
        by=col, ascending=(col != "probability")
    )
    return update_probability_output()


def update_peptide_input(file):
    peptides = "\n".join(pd.read_csv(file, header=None).iloc[:, 0].values)
    return gr.update(value=peptides)


def update_allele_input(file):
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
    with gr.Blocks(title="immunotype", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧬 immunotype")
        gr.Markdown("Peptide-based HLA typing from immunopeptidomics data")

        with gr.Tab("Prediction"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Group():
                        peptide_input = gr.Textbox(
                            label="Peptides input",
                            info="Peptides need to be separated by newlines (example).",
                            lines=20,
                            value=example_peptides,
                        )
                        peptide_file_input = gr.File(label="Peptides input", height=140)
                        _ = gr.ClearButton([peptide_input, peptide_file_input])
                    with gr.Accordion("Additional settings", open=False):
                        with gr.Row():
                            model_toggle = gr.Radio(
                                choices=["Ensemble", "GNN", "Lookup"],
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

            submit_button.click(
                submit,
                inputs=[
                    peptide_input,
                    allele_input,
                    n_peptides_slider,
                    model_toggle,
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

    return demo


# Main function to launch the app via CLI
def main():
    app = create_interface()
    app.launch()


# For direct execution of the app interface
if __name__ == "__main__":
    main()
