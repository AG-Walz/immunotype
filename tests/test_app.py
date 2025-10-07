"""
Tests for the app application.
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

# Add the main folder to path to import from app.py
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from app import create_interface, submit, update_allele_input, update_peptide_input
except ImportError:
    pytest.skip("App dependencies not available. Install with 'pip install immunotype[app]'", allow_module_level=True)


class TestApp:
    """Test cases for the app application."""

    def test_create_interface(self):
        """Test that the Gradio interface can be created."""
        interface = create_interface()
        assert interface is not None
        # Basic check that it's a Gradio interface
        assert hasattr(interface, "launch")

    def test_submit_function(self, monkeypatch):
        """Test the submit function with mock data."""
        # Mock the predict function
        mock_pred_df = pd.DataFrame(
            {
                "allele": ["HLA-A*02:01", "HLA-B*07:02"],
                "probability": [0.8, 0.7],
                "locus": ["A", "B"],
            }
        )
        mock_typing = pd.DataFrame({"allele": ["HLA-A*02:01", "HLA-B*07:02"], "locus": ["A", "B"]})

        def mock_predict(*args, **kwargs):
            return mock_pred_df, mock_typing

        monkeypatch.setattr("app.predict", mock_predict)

        # Test the submit function
        peptides = "ALDGRETD\nASDSGKYL"
        alleles = "HLA-A*02:01\nHLA-B*07:02"
        batch_size = 1000
        use_gnn = True
        use_lookup = True

        result = submit(peptides, alleles, batch_size, use_gnn, use_lookup)

        # Should return a tuple with typing results, probability output, and file update
        assert isinstance(result, tuple)
        assert len(result) == 3

        # First element should be typing results (as numpy array)
        typing_result = result[0]
        assert len(typing_result) > 0

    def test_update_peptide_input(self, tmp_path):
        """Test updating peptide input from file."""
        # Create a temporary peptide file
        peptide_file = tmp_path / "test_peptides.csv"
        test_peptides = pd.DataFrame(["ALDGRETD", "ASDSGKYL"])
        test_peptides.to_csv(peptide_file, index=False, header=False)

        result = update_peptide_input(str(peptide_file))

        # Should return a Gradio update object
        assert hasattr(result, "value") or "value" in result
        if hasattr(result, "value"):
            assert "ALDGRETD" in result.value
        else:
            assert "ALDGRETD" in result["value"]

    def test_update_allele_input(self, tmp_path):
        """Test updating allele input from file."""
        # Create a temporary allele file
        allele_file = tmp_path / "test_alleles.csv"
        test_alleles = pd.DataFrame(["HLA-A*02:01", "HLA-B*07:02"])
        test_alleles.to_csv(allele_file, index=False, header=False)

        result = update_allele_input(str(allele_file))

        # Should return a Gradio update object
        assert hasattr(result, "value") or "value" in result
        if hasattr(result, "value"):
            assert "HLA-A*02:01" in result.value
        else:
            assert "HLA-A*02:01" in result["value"]

    def test_submit_with_invalid_input(self, monkeypatch):
        """Test submit function handles errors gracefully."""

        # Mock predict to raise an exception
        def mock_predict(*args, **kwargs):
            raise ValueError("Test error")

        monkeypatch.setattr("app.predict", mock_predict)

        peptides = "ALDGRETD"
        alleles = "HLA-A*02:01"
        batch_size = 1000
        use_gnn = True
        use_lookup = True

        # This should not crash the application
        with pytest.raises(ValueError):
            submit(peptides, alleles, batch_size, use_gnn, use_lookup)

    def test_peptide_parsing(self, monkeypatch):
        """Test that peptide input parsing works correctly."""
        mock_pred_df = pd.DataFrame(
            {"allele": ["HLA-A*02:01"], "probability": [0.8], "locus": ["A"]}
        )
        mock_typing = pd.DataFrame({"allele": ["HLA-A*02:01"], "locus": ["A"]})

        def mock_predict(peptide_df, *args, **kwargs):
            # Check that peptides were parsed correctly
            expected_peptides = ["ALDGRETD", "ASDSGKYL", "AVDPTSGQ"]
            actual_peptides = peptide_df["peptide"].tolist()
            assert actual_peptides == expected_peptides
            assert all(peptide_df["sample"] == 0)
            return mock_pred_df, mock_typing

        monkeypatch.setattr("app.predict", mock_predict)

        # Test with newline-separated peptides
        peptides = "ALDGRETD\nASDSGKYL\nAVDPTSGQ"
        alleles = "HLA-A*02:01"
        batch_size = 1000
        use_gnn = True
        use_lookup = True

        submit(peptides, alleles, batch_size, use_gnn, use_lookup)
