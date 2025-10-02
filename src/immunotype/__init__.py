"""
Immunotype: Peptide-based HLA typing from immunopeptidomics data.

A modern Python package for predicting HLA typing from peptide sequences
using graph neural networks and lookup tables.
"""

__version__ = "0.0.1"

from .immunotype import predict

__all__ = ["predict"]
