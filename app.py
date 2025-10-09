"""
Gradio app interface for immunotype.

This module provides the app-based user interface for HLA typing predictions.
It creates an interactive Gradio interface that allows users to input peptide
sequences and get HLA typing predictions with visualization.

Usage:
    python app.py  # Launch directly
"""

from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
import pandas as pd
import torch

from immunotype.app import main

# For direct execution of the app interface
if __name__ == "__main__":
    main()
