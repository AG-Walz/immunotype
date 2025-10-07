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
# Basic prediction
immunotype peptides.tsv output.tsv

# With probability details
immunotype peptides.tsv output.tsv --out_probs probabilities.tsv

# Custom settings
immunotype peptides.tsv output.tsv --batch-size 5000 --no-gnn

# Explore all CLI options
immunotype --help
```

### App Interface

**Option 1: Local Gradio App (Recommended)**
```bash
# Install with app dependencies
pip install immunotype[app]

# Run the Gradio app
python app.py
```

**Option 2: Hugging Face Spaces**  
The app is Hugging Face Spaces compatible. Simply upload `app.py` and the `src/` folder to your Space.

### Python API

```python
import pandas as pd
from immunotype import predict

# Prepare data
peptides = pd.DataFrame({
    'peptide': ['ALDGRETD', 'ASDSGKYL'],
    'sample': ['sample1', 'sample1']
})

alleles = ['HLA-A*02:01', 'HLA-B*07:02', 'HLA-C*07:02']

# Make predictions
predictions, typing = predict(
    peptide_df=peptides,
    selected_alleles=alleles
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