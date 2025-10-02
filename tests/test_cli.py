"""
Tests for the CLI module.
"""

import pytest
from click.testing import CliRunner
import pandas as pd
from pathlib import Path

from immunotype.cli import main


class TestCLI:
    """Test cases for the CLI interface."""

    def test_cli_help(self):
        """Test that help message is displayed correctly."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'Predict HLA typing from immunopeptide sequences' in result.output

    def test_cli_basic_functionality(self, peptide_file, allele_file, output_file, monkeypatch):
        """Test basic CLI functionality with mock data."""
        # Mock the predict function to avoid heavy computation
        def mock_predict(*args, **kwargs):
            # Return mock prediction results
            pred_df = pd.DataFrame({
                'sample': ['0', '0', '0'],
                'locus': ['A', 'B', 'C'],
                'allele': ['HLA-A*02:01', 'HLA-B*07:02', 'HLA-C*07:02'],
                'probability': [0.8, 0.7, 0.9]
            })
            typing = pd.DataFrame({
                'sample': ['0', '0'],
                'allele': ['HLA-A*02:01', 'HLA-B*07:02']
            })
            return pred_df, typing
        
        monkeypatch.setattr("immunotype.cli.predict", mock_predict)
        
        runner = CliRunner()
        result = runner.invoke(main, [
            str(peptide_file),
            str(output_file),
            '--hla-input', str(allele_file),
            '--no-gnn'  # Use only lookup to speed up testing
        ])
        
        assert result.exit_code == 0
        assert 'Predicted HLA typing' in result.output
        assert output_file.exists()

    def test_cli_invalid_input(self):
        """Test CLI with invalid input file."""
        runner = CliRunner()
        result = runner.invoke(main, ['nonexistent_file.tsv'])
        assert result.exit_code != 0

    def test_cli_both_methods_disabled(self, peptide_file, allele_file):
        """Test CLI fails when both GNN and lookup are disabled."""
        runner = CliRunner()
        result = runner.invoke(main, [
            str(peptide_file),
            'dummy_output.tsv',
            '--hla-input', str(allele_file),
            '--no-gnn',
            '--no-lookup'
        ])
        assert result.exit_code != 0
        assert 'Cannot disable both GNN and lookup methods' in result.output

    def test_cli_custom_batch_size(self, peptide_file, allele_file, monkeypatch):
        """Test CLI with custom batch size."""
        def mock_predict(*args, **kwargs):
            assert kwargs.get('batch_size') == 5
            return pd.DataFrame(), pd.DataFrame({'sample': [], 'allele': []})
        
        monkeypatch.setattr("immunotype.cli.predict", mock_predict)
        
        runner = CliRunner()
        result = runner.invoke(main, [
            str(peptide_file),
            'dummy_output.tsv',
            '--hla-input', str(allele_file),
            '--batch-size', '5',
            '--no-gnn'
        ])
        assert result.exit_code == 0

    def test_cli_max_peptides_option(self, peptide_file, allele_file, monkeypatch):
        """Test CLI with max peptides option."""
        def mock_predict(*args, **kwargs):
            assert kwargs.get('max_n_peptides') == 5000
            return pd.DataFrame(), pd.DataFrame({'sample': [], 'allele': []})
        
        monkeypatch.setattr("immunotype.cli.predict", mock_predict)
        
        runner = CliRunner()
        result = runner.invoke(main, [
            str(peptide_file),
            'dummy_output.tsv',
            '--hla-input', str(allele_file),
            '--max-n-peptides', '5000',
            '--no-gnn'
        ])
        assert result.exit_code == 0