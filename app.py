"""
Gradio app interface for immunotype.

This module provides the app-based user interface for HLA typing predictions.
It creates an interactive Gradio interface that allows users to input peptide
sequences and get HLA typing predictions with visualization.

Usage:
    python app.py  # Launch directly
"""

from immunotype.app import main

# For direct execution of the app interface
if __name__ == "__main__":
    main()
