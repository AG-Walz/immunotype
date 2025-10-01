import numpy as np

# B, Z, X, J are ambiguous amino acids
# ? is an unknown amino acid
AMINO_ACIDS = np.array([
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M',
    'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'B', 'Z', 'X', 'J', '?'
])
SEQUENCE_TOKENS = np.array([
    '[CLS]', '[SEP]', '[MASK]'
])
PLACEHOLDERS = np.array([
    'HLA-A*homozygous', 'HLA-B*homozygous', 'HLA-C*homozygous'
])
TOKENS = np.concatenate([SEQUENCE_TOKENS, AMINO_ACIDS, PLACEHOLDERS])
TOKEN_VOCABULARY = dict(zip(TOKENS, range(1, len(TOKENS) + 1)))


LOOKUP_HOMOZYGOUS_THRESHOLDS = {'A': 0.425, 'B': 0.3, 'C': 0.4}
ENSEMBLE_MODEL_WEIGHTS = {'A': 0.6, 'B': 0.4, 'C': 0.7}