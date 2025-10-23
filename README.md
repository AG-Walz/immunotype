---
title: immunotype
emoji: 🧬
colorFrom: purple
colorTo: indigo
sdk: gradio
python_version: 3.10.0
app_file: app.py
license: mit
---

# 🧬 immunotype

**Peptide-based HLA typing from immunopeptidomics data**

[![CI](https://github.com/immunotype/immunotype/workflows/CI/badge.svg)](https://github.com/immunotype/immunotype/actions?query=workflow%3ACI)
[![PyPI version](https://badge.fury.io/py/immunotype.svg)](https://badge.fury.io/py/immunotype)
[![Python versions](https://img.shields.io/pypi/pyversions/immunotype.svg)](https://pypi.org/project/immunotype/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Immunotype is a modern Python package for predicting HLA typing from peptide sequences using graph neural networks and lookup tables. It combines machine learning approaches with high-performance lookup methods to provide accurate and fast HLA allele predictions from immunopeptidomics data.


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
│    --prob_output       PATH                   Save detailed HLA probabilities to specified TSV file.                                                                    │
│    --hla-input         PATH                   Path to the HLA input file containing alleles to consider. [default: <immunotype-package-path>/data/selected_alleles.csv] │
│    --max-n-peptides    INTEGER                Maximum number of peptides to predict at once. [default: 50000]                                                           │
│    --batch_size        INTEGER                How many samples should be predicted simultaneously. [default: 1]                                                         │
│    --prediction_model  [ensemble|gnn|lookup]  Select which model to use. [default: ensemble]                                                                            │
│    --use_gpu                                  Run prediction on GPU instead of CPU.                                                                                     │
│    --version                                  Show the version and exit.                                                                                                │
│    --help                                     Show this message and exit.                                                                                               │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

 For more information, visit: https://github.com/AG-Walz/immunotype
 ```

```bash
# Basic prediction
immunotype src/examples/single_sample_input.tsv test_single_sample_typing.tsv

# With probability details
immunotype src/examples/single_sample_input.tsv test_single_sample_typing.tsv --out_probs test_single_sample_probabilities.tsv

# Custom settings
immunotype src/examples/single_sample_input.tsv test_single_sample_typing.tsv --batch-size 100 --prediction_model gnn

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

## 📊 Input Formats

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


## 📝 Authors

- Matteo Pilz
- Jonas Scheid


## 📚 Citation

```bibtex
@software{immunotype,
  title={immunotype: Peptide-based HLA typing from immunopeptidomics data},
  author={Pilz, Matteo and Scheid, Jonas},
  url={https://github.com/AG-Walz/immunotype},
  year={2025}
}
```
