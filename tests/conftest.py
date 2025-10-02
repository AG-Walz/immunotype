"""
Test configuration and fixtures for immunotype.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import os

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_peptides():
    """Sample peptide dataframe for testing."""
    return pd.DataFrame({
        'peptide': ['ALDGRETD', 'ASDSGKYL', 'AVDPTSGQ', 'DISQTSKY', 'DSDINNRL'],
        'sample': ['0', '0', '0', '0', '0']
    })


@pytest.fixture
def sample_alleles():
    """Sample HLA alleles for testing."""
    return ['HLA-A*02:01', 'HLA-B*07:02', 'HLA-C*07:02']


@pytest.fixture
def peptide_file(tmp_path, sample_peptides):
    """Temporary peptide file for CLI testing."""
    file_path = tmp_path / "test_peptides.tsv"
    sample_peptides.to_csv(file_path, sep='\t', index=False, header=False)
    return file_path


@pytest.fixture
def allele_file(tmp_path, sample_alleles):
    """Temporary allele file for CLI testing."""
    file_path = tmp_path / "test_alleles.csv"
    pd.DataFrame(sample_alleles).to_csv(file_path, index=False, header=False)
    return file_path


@pytest.fixture
def output_file(tmp_path):
    """Temporary output file for CLI testing."""
    return tmp_path / "test_output.tsv"


@pytest.fixture
def mock_model_weights(tmp_path):
    """Mock model weights file for testing."""
    weights_file = tmp_path / "mock_weights.pth"
    weights_file.write_text("mock weights")
    return weights_file


class MockModel:
    """Mock GNN model for testing."""
    
    def eval(self):
        pass
    
    def __call__(self, batch):
        import torch
        import numpy as np
        # Return mock predictions
        batch_size = len(batch.sample) if hasattr(batch, 'sample') else 1
        num_alleles = 3  # Mock number of alleles
        return torch.tensor(np.random.random((batch_size, num_alleles)))


@pytest.fixture
def mock_gnn_model(monkeypatch):
    """Mock the GNN model for testing."""
    mock_model = MockModel()
    
    def mock_load_weights(model, path):
        return mock_model
    
    monkeypatch.setattr("immunotype.utils.load_weights", mock_load_weights)
    return mock_model