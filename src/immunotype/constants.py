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

# number of decimals shown and in export
DECIMAL_PRECISION = 4

LOOKUP_HOMOZYGOUS_THRESHOLDS = {"A": 0.425, "B": 0.3, "C": 0.325}
ENSEMBLE_GNN_WEIGHTS = {"A": 0.7, "B": 0.8, "C": 0.8}

# MHC_SEQUENCE_DF stores the MHC sequences together with their identifier
MHC_SEQUENCE_DF = pd.read_csv(PACKAGE_ROOT / "data" / "mhc_sequences.csv")

# LOOKUP_Df stores all unique peptide-HLA combinations with their locus
LOOKUP_DF = pd.read_csv(PACKAGE_ROOT / "data" / "lookup_db.csv")

PREDICTION_MODELS = ["Ensemble", "GNN", "Lookup"]

APP_HELP_SECTION = """
## Documentation

### Usage
1. Enter peptide sequences (one per line) or upload a TSV file.
2. Click **Run prediction**.
3. View results in the **Typing** table; download the full probability matrix as TSV.

### Input Formats
- **Single sample**: One peptide sequence per line (e.g., `ALDGRETD`).
- **Multiple samples**: TSV with `sample` and `peptide` columns.
- **HLA alleles**: One allele per line (e.g., `HLA-A*24:27`).

### Output
- **Typing**: Predicted HLA alleles for your sample.
- **Probabilities**: Detailed probability scores for all tested alleles.

### Advanced Settings
- **Prediction model**: Ensemble (GNN + lookup), or either alone.
- **HLA alleles**: Alleles included in prediction. Changing this is not recommended as it may reduce accuracy.
- **Max peptides per batch**: Should be as high as memory allows. Combined with batch size, controls memory usage (especially on GPU).
- **Batch size**: Increasing batch size generally speeds up CPU prediction. On GPU, keep it lower.
- **Use GPU**: Run prediction on GPU instead of CPU.

Max peptides, batch size, and GPU settings do not affect lookup-only predictions.

### Citation
If you use immunotype in your research, please cite TODO.
"""
