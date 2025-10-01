import itertools
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from model import GNN
import gradio as gr
import seaborn as sns
from immunotype import *


cm = sns.light_palette("green", as_cmap=True)
DEVICE = torch.device('cpu')

probability_df = None


def submit(peptides, alleles, batch_size):
    global probability_df

    peptide_df = pd.DataFrame(peptides.replace('\n', ',').split(','), columns=['peptide'])
    peptide_df['sample'] = 0
    alleles = pd.Series(alleles.replace('\n', ',').split(','))
    probability_df, typing = predict(
        peptide_df,
        alleles,
        use_gnn=True,
        use_lookup=True,
        batch_size=10,
        max_n_peptides=10_000,
        gnn_weight_path="weights/gnn_model_weights.pth",
    )
    typing = typing[['allele', 'locus']].values
    csv_path = 'probabilities.csv'
    probability_df.to_csv(csv_path, index=False)
    return (
        typing, update_probability_output(),
        gr.update(value=csv_path, visible=True)
    )


def update_probability_output():
    global probability_df
    style = probability_df.style.background_gradient(cmap=cm)
    return style


def sort_table(col):
    global probability_df
    probability_df = probability_df.sort_values(by=col, ascending=(col == 'allele'))
    return update_probability_output()


def update_peptide_input(file):
    peptides = '\n'.join(pd.read_csv(file, header=None).iloc[:, 0].values)
    return gr.update(value=peptides)

def update_allele_input(file):
    alleles = '\n'.join(pd.read_csv(file, header=None).iloc[:, 0].values)
    return gr.update(value=alleles)


example_peptides = \
    'ALDGRETD\n' \
    'ASDSGKYL\n' \
    'AVDPTSGQ\n' \
    'DISQTSKY\n' \
    'DSDINNRL'

example_alleles =  '\n'.join(pd.read_csv('data/selected_alleles.csv', header=None)[0].values)


with gr.Blocks() as demo:
    gr.Markdown("# immunotype")
    gr.Markdown("Peptide-based HLA typing from immunopeptidomics data")
    with gr.Tab("Prediction"):
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Group():
                    peptide_input = gr.Textbox(
                        label="Peptides input",
                        info="Peptides need to be separated by newlines (example).",
                        lines=20,
                        value=example_peptides
                    )
                    peptide_file_input = gr.File(
                        label="Peptides input",
                        height=140
                    )
                with gr.Accordion("Additional settings", open=False):
                    with gr.Group():
                        allele_input = gr.Textbox(
                            label="HLA allele input",
                            info="Alleles need to be separated by newlines (default).",
                            lines=20,
                            value=example_alleles
                        )
                        allele_file_input = gr.File(
                            label="HLA allele input",
                            height=140
                        )
                    batch_size_slider = gr.Slider(
                        1_000, 100_000,
                        value=10_000,
                        step=1_000,
                        interactive=True,
                        label="Batch size",
                        info='Controls the maximum number of peptides per prediction run. '
                             'Note that all peptides are predicted and allele probabilities averageg, '
                             'if the number of peptides is larger than the batch size'
                    )

                submit_button = gr.Button("Submit")
            with gr.Column(scale=4):
                typing_output = gr.HighlightedText(
                    label="Typing",
                )
                with gr.Group():
                    col_selector = gr.Dropdown(choices=["allele", "probability"], label="Sort by")
                    typing_probabilities = gr.Dataframe(
                        headers=["allele", "probability", "locus"],
                        datatype=["str", "number", "str"],
                        row_count=1,
                        label="Typing probabilities",
                        col_count=(3, "fixed"),
                        show_copy_button=True,
                        show_search='search',
                    )
                    file_output = gr.File(label="Download CSV", visible=False)

        submit_button.click(
            submit,
            inputs=[peptide_input, allele_input, batch_size_slider],
            outputs=[
                typing_output, typing_probabilities,
                file_output
            ]
        )
        peptide_file_input.upload(update_peptide_input, inputs=peptide_file_input, outputs=peptide_input)
        allele_file_input.upload(update_allele_input, inputs=allele_file_input, outputs=allele_input)
        col_selector.change(sort_table, inputs=col_selector, outputs=typing_probabilities)

    with gr.Tab("Help"):
        gr.Markdown(
            "Tutorial, resources and sources"
        )




if __name__ == '__main__':
    demo.launch()