"""
Gradio app interface for immunotype.

This module provides the app-based user interface for HLA typing predictions.
It creates an interactive Gradio interface that allows users to input peptide
sequences and get HLA typing predictions with visualization.

Usage:
    python app.py  # Launch directly
"""

import base64
import html
import re
import tempfile
import warnings
from pathlib import Path

import gradio as gr
import pandas as pd

from immunotype.constants import APP_HELP_SECTION, DECIMAL_PRECISION, PREDICTION_MODELS
from immunotype.immunotype import predict
from immunotype.utils import parse_allele_input, parse_peptide_input

# Get package root directory
PACKAGE_ROOT = Path(__file__).parent

# Base64-encode logos for inline use (no file serving needed)
_logo_dark_path = PACKAGE_ROOT / "assets" / "immunotype_logo_dark_transparent.png"
_logo_dark_b64 = base64.b64encode(_logo_dark_path.read_bytes()).decode() if _logo_dark_path.exists() else ""
_logo_light_path = PACKAGE_ROOT / "assets" / "immunotype_logo_light_transparent.png"
_logo_light_b64 = base64.b64encode(_logo_light_path.read_bytes()).decode() if _logo_light_path.exists() else ""

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.slate,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    radius_size=gr.themes.sizes.radius_sm,
).set(
    block_border_width="1px",
    block_label_text_size="*text_md",
    block_title_text_size="*text_md",
    block_shadow="none",
)


def _style_probabilities(df):
    """Apply consistent styling to probability DataFrame."""
    return df.style.format(precision=DECIMAL_PRECISION).background_gradient(
        cmap="YlGnBu", vmin=0, vmax=1
    )


def submit(
    peptides: str,
    alleles: str,
    max_n_peptides: int,
    batch_size: int,
    prediction_model: str,
    use_gpu: bool,
    progress=gr.Progress(),
):
    """Executes the script by pressing the submit button."""
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
                progress=progress.tqdm,
            )
            for warning in w:
                gr.Warning(html.escape(str(warning.message)), duration=None)
    except Exception as e:
        raise gr.Error(html.escape(str(e)), duration=None)

    # samples = typing_df["sample"].astype(str).tolist()
    # sample_tag = re.sub(r"[^\w\-]", "_", "_".join(samples))[:100]
    tmpdir = tempfile.gettempdir()

    # typing_path = Path(tmpdir) / f"{sample_tag}_typing.tsv"
    typing_path = Path(tmpdir) / "typing_output.tsv"
    typing_df.to_csv(typing_path, index=False, sep="\t")

    # prob_path = Path(tmpdir) / f"{sample_tag}_probabilities.tsv"
    prob_path = Path(tmpdir) / "probability_output.tsv"
    probability_df.to_csv(
        prob_path,
        index=False,
        float_format=f"%.{DECIMAL_PRECISION}f",
        sep="\t",
    )

    return (
        typing_df,
        _style_probabilities(probability_df),
        gr.File(value=str(typing_path), visible=True),
        gr.File(value=str(prob_path), visible=True),
        probability_df,
    )


def sort_table(col: str, probability_df: pd.DataFrame):
    """Sort the table by col."""
    if probability_df is None or probability_df.empty:
        return gr.Dataframe(), probability_df
    probability_df = probability_df.sort_values(
        by=col, ascending=(col != "probability")
    )
    return _style_probabilities(probability_df), probability_df


def update_peptide_input(file: str):
    """Update the peptide input shown in the text field."""
    peptides = "\n".join(pd.read_csv(file, header=None).iloc[:, 0].values)
    return gr.Textbox(value=peptides)


def update_allele_input(file: str):
    """Update the allele input shown in the text field."""
    alleles = "\n".join(pd.read_csv(file, header=None).iloc[:, 0].values)
    return gr.Textbox(value=alleles)


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
    with gr.Blocks(
        title="immunotype",
    ) as app:
        gr.HTML(
            f"<div style='display: flex; align-items: center; gap: 16px; margin-bottom: 8px;'>"
            f"<img class='logo-dark' src='data:image/png;base64,{_logo_dark_b64}' "
            f"style='height: 48px;' alt='immunotype'>"
            f"<img class='logo-light' src='data:image/png;base64,{_logo_light_b64}' "
            f"style='height: 48px;' alt='immunotype'>"
            f"<span style='color: var(--body-text-color-subdued); font-size: 0.9rem;'>"
            f"Peptide-based HLA typing from immunopeptidomics data</span>"
            f"</div>"
        )

        prob_state = gr.State(value=None)

        with gr.Tab("Prediction"):
            with gr.Row():
                with gr.Column(scale=1, min_width=360):
                    with gr.Group():
                        peptide_input = gr.Textbox(
                            label="Peptide sequences",
                            info="One peptide per line, or upload a file below.",
                            lines=16,
                            max_lines=20,
                            value=example_peptides,
                            placeholder="VLRGAIETY\nEENTLVQNY\n...",  # shown when input is empty
                        )
                        peptide_file_input = gr.File(label="Peptide file", height=140)
                        _ = gr.ClearButton([peptide_input, peptide_file_input])
                    with gr.Accordion("Advanced settings", open=False):
                        with gr.Row():
                            model_toggle = gr.Radio(
                                choices=PREDICTION_MODELS,
                                value="Ensemble",
                                label="Prediction model",
                                info="Ensemble uses both the GNN and the lookup table. "
                                + "Alternatively, use either alone.",
                            )
                        with gr.Group():
                            allele_input = gr.Textbox(
                                label="HLA alleles",
                                info="One allele per line. Changing this is not recommended.",
                                lines=20,
                                value=example_alleles,
                            )
                            allele_file_input = gr.File(
                                label="HLA allele file", height=140
                            )
                            _ = gr.ClearButton([allele_input, allele_file_input])
                        n_peptides_slider = gr.Slider(
                            1_000,
                            100_000,
                            value=50_000,
                            step=1_000,
                            interactive=True,
                            label="Max peptides per batch",
                            info="All peptides are predicted; probabilities are averaged "
                            + "if a sample exceeds this limit.",
                        )
                        batch_size_slider = gr.Slider(
                            1,
                            100,
                            value=1,
                            step=1,
                            interactive=True,
                            label="Batch size",
                            info="Number of samples predicted simultaneously. "
                            + "Affects only Ensemble and GNN modes.",
                        )
                        use_gpu = gr.Checkbox(
                            label="Use GPU", info="Predict on GPU instead of CPU."
                        )

                    submit_button = gr.Button(
                        "Run prediction", variant="primary", size="lg"
                    )

                with gr.Column(scale=3, min_width=500):
                    with gr.Group():
                        typing = gr.Dataframe(
                            headers=["sample", "typing"],
                            datatype=["str", "str"],
                            row_count=1,
                            column_count=(2, "fixed"),
                            buttons=["copy"],
                            label="Typing",
                        )
                    with gr.Group():
                        col_selector = gr.Dropdown(
                            choices=["sample", "allele", "probability"],
                            label="Sort by",
                            scale=1,
                            min_width=160,
                        )
                        typing_probabilities = gr.Dataframe(
                            headers=["sample", "allele", "probability", "locus"],
                            datatype=["str", "str", "number", "str"],
                            row_count=1,
                            label="Typing probabilities",
                            column_count=(4, "fixed"),
                            buttons=["copy"],
                        )
                    with gr.Row():
                        typing_output = gr.File(label="Typing results", visible=False)
                        probability_output = gr.File(
                            label="Probability matrix", visible=False
                        )

            submit_button.click(
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
                    prob_state,
                ],
            )
            peptide_file_input.upload(
                update_peptide_input, inputs=peptide_file_input, outputs=peptide_input
            )
            allele_file_input.upload(
                update_allele_input, inputs=allele_file_input, outputs=allele_input
            )
            col_selector.change(
                sort_table,
                inputs=[col_selector, prob_state],
                outputs=[typing_probabilities, prob_state],
            )
        with gr.Tab("Help"):
            gr.Markdown(APP_HELP_SECTION)

    return app


# Main function to launch the app via CLI
def main():
    app = create_interface()
    app.launch(
        theme=theme,
        css=".logo-dark { display: none; } .logo-light { display: inline; } "
        ".dark .logo-dark { display: inline; } .dark .logo-light { display: none; }",
    )


# For direct execution of the app interface
if __name__ == "__main__":
    main()
