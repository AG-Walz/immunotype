<img src="assets/immunotype_logo_dark_transparent.png" alt="immunotype logo" width="400">

**Peptide-based HLA typing from immunopeptidomics data**

[![CI](https://github.com/immunotype/immunotype/workflows/CI/badge.svg)](https://github.com/immunotype/immunotype/actions?query=workflow%3ACI)
[![PyPI version](https://badge.fury.io/py/immunotype.svg)](https://badge.fury.io/py/immunotype)
[![Python versions](https://img.shields.io/pypi/pyversions/immunotype.svg)](https://pypi.org/project/immunotype/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

immunotype predicts HLA class I alleles directly from immunopeptidomics data — no separate HLA typing experiment needed. immunotype combines a graph neural network with a curated mono-allelic lookup table in an ensemble model, achieving **87.2% accuracy** at protein-level resolution across diverse human tissues.

## 🚀 Quick Start

### Installation

```bash
# Core CLI functionality
pip install immunotype

# With app interface (optional)
pip install immunotype[app]

# Development dependencies
pip install immunotype[dev]

# Everything
pip install immunotype[all]
```

### Command Line Interface

```bash

 Usage: immunotype [OPTIONS] PEPTIDE_INPUT TYPING_OUTPUT

 Predict HLA typing from immunopeptide sequences.
 This tool uses graph neural networks and lookup tables to predict HLA allele typing from immunopeptidomics data. Provide a peptide input file and optionally customize the HLA alleles to consider.

╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  PEPTIDE_INPUT       PATH                   TSV input file. Either a single column of peptides or two columns with sample IDs and peptides. [required]                │
│ *  TYPING_OUTPUT       PATH                   Path to save the typing output. [required]                                                                                │
│    --prob-output       PATH                   Save detailed HLA probabilities to specified TSV file.                                                                    │
│    --hla-input         PATH                   Path to the HLA input file containing alleles to consider. [default: <immunotype-package-path>/data/selected_alleles.csv] │
│    --max-n-peptides    INTEGER                Maximum number of peptides to predict at once. [default: 50000]                                                           │
│    --batch-size        INTEGER                How many samples should be predicted simultaneously. [default: 1]                                                         │
│    --prediction-model  [ensemble|gnn|lookup]  Select which model to use. [default: ensemble]                                                                            │
│    --use-gpu                                  Run prediction on GPU instead of CPU.                                                                                     │
│    --version                                  Show the version and exit.                                                                                                │
│    --help                                     Show this message and exit.                                                                                               │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

 For more information, visit: https://github.com/AG-Walz/immunotype
 ```

```bash
# Basic prediction
immunotype src/immunotype/examples/single_sample_input.tsv test_single_sample_typing.tsv

# With probability details
immunotype src/immunotype/examples/single_sample_input.tsv test_single_sample_typing.tsv --prob-output test_single_sample_probabilities.tsv

# Custom settings
immunotype src/immunotype/examples/single_sample_input.tsv test_single_sample_typing.tsv --batch-size 100 --prediction-model gnn

# Explore all CLI options
immunotype --help
```

### App Interface

```bash
# Install with app dependencies
pip install immunotype[app]

# Run the Gradio app
immunotype-app
```

### Python API

```python
import pandas as pd
from immunotype import predict

# Prepare data
peptides = pd.DataFrame({
    'peptide': ['ALDGRETD', 'ASDSGKYL'],
    'sample': ['sample1', 'sample1']
})

alleles = pd.DataFrame({
    'allele': ['HLA-A*02:01', 'HLA-B*07:02', 'HLA-C*07:02']
})

# Make predictions
predictions, typing = predict(
    peptide_df=peptides,
    allele_df=alleles
)
```

## 📥 Input Formats

**Single-column peptides:**
```
ALDGRETD
ASDSGKYL
AVDPTSGQ
```

**Multi-sample format:**
```
sample	peptide
1	ALDGRETD
1	ASDSGKYL
2	AVDPTSGQ
```


## 📤 Output

**Typing (default):** Top 2 alleles per locus (A, B, C) per sample.
```
sample	typing
sample_0	HLA-A*32:01;HLA-A*68:01;HLA-B*15:01;HLA-B*44:02;HLA-C*03:03;HLA-C*07:04
```

**Probabilities (`--prob-output`):** Per-allele scores from each prediction mode.
```
sample	locus	allele	probability_gnn	probability_lookup	probability
sample_0	A	HLA-A*01:01	0.0005	0.0000	0.0003
sample_0	A	HLA-A*02:01	0.0000	0.1945	0.0584
```

## 📝 Authors

- Matteo Pilz
- Jonas Scheid


## 📚 Citation

```bibtex
@software{immunotype,
  title={immunotype: Peptide-based HLA typing from immunopeptidomics data},
  author={Pilz, Matteo and Scheid, Jonas},
  url={https://github.com/AG-Walz/immunotype},
  year={2026}
}
```
