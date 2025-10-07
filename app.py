"""
Gradio app interface for immunotype.

This module provides the app-based user interface for HLA typing predictions.
It creates an interactive Gradio interface that allows users to input peptide
sequences and get HLA typing predictions with visualization.

Usage:
    python app.py  # Launch directly
"""

from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
import pandas as pd
import torch

from immunotype.immunotype import predict

# Get package root directory
PACKAGE_ROOT = Path(__file__).parent / "src" / "immunotype"

cm = gr.themes.Soft().primary_hue
DEVICE = torch.device("cpu")

probability_df = None


def submit(peptides, alleles, batch_size, use_gnn, use_lookup):
    global probability_df

    peptide_df = pd.DataFrame(peptides.replace("\n", ",").split(","), columns=["peptide"])
    peptide_df["sample"] = 0
    alleles = pd.Series(alleles.replace("\n", ",").split(","))
    probability_df, typing = predict(
        peptide_df,
        alleles,
        use_gnn=use_gnn,
        use_lookup=use_lookup,
        batch_size=10,
        max_n_peptides=10_000,
        gnn_weight_path=PACKAGE_ROOT / "weights" / "gnn_model_weights.pth",
    )
    typing = typing[["allele", "locus"]].values
    csv_path = "probabilities.csv"
    probability_df.to_csv(csv_path, index=False)
    return (typing, update_probability_output(), gr.update(value=csv_path, visible=True))


def update_probability_output():
    global probability_df
    style = probability_df.style.background_gradient(cmap=cm)
    return style


def sort_table(col):
    global probability_df
    probability_df = probability_df.sort_values(by=col, ascending=(col == "allele"))
    return update_probability_output()


def update_peptide_input(file):
    peptides = "\n".join(pd.read_csv(file, header=None).iloc[:, 0].values)
    return gr.update(value=peptides)


def update_allele_input(file):
    alleles = "\n".join(pd.read_csv(file, header=None).iloc[:, 0].values)
    return gr.update(value=alleles)


example_peptides = "ALDGRETD\nASDSGKYL\nAVDPTSGQ\nDISQTSKY\nDSDINNRL"

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
                    with gr.Accordion("Additional settings", open=False):
                        with gr.Row():
                            use_gnn_toggle = gr.Checkbox(
                                label="Use GNN model",
                                value=True,
                                info="Enable/disable the pre-trained graph neural network model"
                            )
                            use_lookup_toggle = gr.Checkbox(
                                label="Use lookup table",
                                value=True,
                                info="Enable/disable the peptide-HLA lookup table"
                            )
                        with gr.Group():
                            allele_input = gr.Textbox(
                                label="HLA allele input",
                                info="Alleles need to be separated by newlines.",
                                lines=20,
                                value=example_alleles,
                            )
                            allele_file_input = gr.File(label="HLA allele input", height=140)
                        batch_size_slider = gr.Slider(
                            1_000,
                            100_000,
                            value=10_000,
                            step=1_000,
                            interactive=True,
                            label="Batch size",
                            info="Controls the maximum number of peptides per prediction run. "
                            "Note that all peptides are predicted and allele probabilities averaged, "
                            "if the number of peptides is larger than the batch size",
                        )

                    submit_button = gr.Button("Submit", variant="primary")

                with gr.Column(scale=4):
                    typing_output = gr.HighlightedText(
                        label="Typing",
                    )
                    with gr.Group():
                        col_selector = gr.Dropdown(
                            choices=["allele", "probability"], label="Sort by"
                        )
                        typing_probabilities = gr.Dataframe(
                            headers=["allele", "probability", "locus"],
                            datatype=["str", "number", "str"],
                            row_count=1,
                            label="Typing probabilities",
                            col_count=(3, "fixed"),
                            show_copy_button=True,
                        )
                        file_output = gr.File(label="Download CSV", visible=False)

            submit_button.click(
                submit,
                inputs=[peptide_input, allele_input, batch_size_slider, use_gnn_toggle, use_lookup_toggle],
                outputs=[typing_output, typing_probabilities, file_output],
            )
            peptide_file_input.upload(
                update_peptide_input, inputs=peptide_file_input, outputs=peptide_input
            )
            allele_file_input.upload(
                update_allele_input, inputs=allele_file_input, outputs=allele_input
            )
            col_selector.change(sort_table, inputs=col_selector, outputs=typing_probabilities)
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


# For direct execution of the app interface
if __name__ == "__main__":
    app = create_interface()
    app.launch()
