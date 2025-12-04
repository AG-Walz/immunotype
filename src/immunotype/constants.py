from pathlib import Path

import numpy as np
import pandas as pd

# ASCII art banner for the application
ASCII_BANNER = """    _                                       __
   (_)___ ___  ____ ___  __  ______  ____  ╱ ╱___  ______  ___
  ╱ ╱ __ `__ ╲╱ __ `__ ╲╱ ╱ ╱ ╱ __ ╲╱ __ ╲╱ __╱ ╱ ╱ ╱ __ ╲╱ _ ╲
 ╱ ╱ ╱ ╱ ╱ ╱ ╱ ╱ ╱ ╱ ╱ ╱ ╱_╱ ╱ ╱ ╱ ╱ ╱_╱ ╱ ╱_╱ ╱_╱ ╱ ╱_╱ ╱  __╱
╱_╱_╱ ╱_╱ ╱_╱_╱ ╱_╱ ╱_╱╲__,_╱_╱ ╱_╱╲____╱╲__╱╲__, ╱ .___╱╲___╱
                                            ╱____╱_╱           """

# Authors information
__authors__ = ["Matteo Pilz", "Jonas Scheid"]

# Get package root directory
PACKAGE_ROOT = Path(__file__).parent

# B, Z, X, J are ambiguous amino acids
# ? is an unknown amino acid
AMINO_ACIDS = np.array(
    [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
        "B",
        "Z",
        "X",
        "J",
        "?",
    ]
)
SEQUENCE_TOKENS = np.array(["[CLS]", "[SEP]", "[MASK]"])
PLACEHOLDERS = np.array(["HLA-A*homozygous", "HLA-B*homozygous", "HLA-C*homozygous"])
TOKENS = np.concatenate([SEQUENCE_TOKENS, AMINO_ACIDS, PLACEHOLDERS])
TOKEN_VOCABULARY = dict(zip(TOKENS, range(1, len(TOKENS) + 1), strict=True))


LOOKUP_HOMOZYGOUS_THRESHOLDS = {"A": 0.55, "B": 0.4, "C": 0.325}
ENSEMBLE_GNN_WEIGHTS = {"A": 0.5, "B": 0.5, "C": 0.9}

# MHC_SEQUENCE_DF stores the MHC sequences together with their identifier
MHC_SEQUENCE_DF = pd.read_csv(PACKAGE_ROOT / "data" / "mhc_sequences.csv")

# LOOKUP_Df stores all unique peptide-HLA combinations with their locus
LOOKUP_DF = pd.read_csv(PACKAGE_ROOT / "data" / "lookup_db.csv")

PREDICTION_MODELS = ["Ensemble", "GNN", "Lookup"]

APP_HELP_SECTION = """
## 📚 Tutorial and Resources

### Usage
1. **Peptides input**: Enter peptide sequences separated by newlines, or upload a file.
2. **Submit**: Click submit to run the prediction.
3. **View results**: See predicted typing and download detailed probabilities.

### Input Formats
#### Peptides
- **One sample**: One peptide sequence per line (e.g., `ALDGRETD`).
- **Multiple samples**: Header with sample & peptide, then sample id and peptide sequence.
- **HLA alleles**: One allele per line (e.g., HLA-A*24-27).
- **Files**: TSV/CSV files with either peptide sequences only or with: sample, peptide as columns.

### Output
- **Typing**: Predicted HLA alleles for your sample.
- **Probabilities**: Detailed probability scores for all tested alleles.
- **TSV Download**: Full results table for further analysis.

### Additional Settings
- **Select which model to use**: You can choose between ensemble or GNN and lookup only.
- **HLA allele input**: Choose the alleles included in the prediction (it is not recommended to change this, see below).¹
- **Maximum number of peptides**: Change the maximum number of peptides that should be predicted simultaneously.²
- **Batch size**: Number of samples that should be predicted simultaneously.³
- **Use GPU**: Run prediction on GPU, will throw a warning if activated and no GPU is available.

### Remarks
¹ It is not recommended to change the selected alleles, as there is a high probability it will influence the prediction accuracy.\n
² The maximum number of peptides should be as high as possible and ideally larger than the number of peptides in each sample.
Together with the batch size, this setting can be used to prevent memory overflow, especially on GPU.\n
³ In most cases and especially on CPU, the batch size can be increased to speed up the prediction.
When running on GPU, it is advised to keep the batch size lower.\n
\n
Maximum number of peptides, batch size and usage of GPU do not affect a lookup only prediction.

### Citation
If you use immunotype in your research, please cite TODO.
"""
