"""
Tests for the core immunotype functionality.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest

from immunotype.immunotype import predict
from immunotype.immunotype import predict_lookup
from immunotype.immunotype import prepare_data


class TestImmunotype:
    """Test cases for core immunotype functionality."""

    def test_predict_lookup_basic(self, sample_peptides, sample_alleles, monkeypatch):
        """Test basic lookup prediction functionality."""
        # Mock lookup database
        mock_lookup_db = pd.DataFrame(
            {
                "peptide": ["ALDGRETD", "ASDSGKYL", "AVDPTSGQ"],
                "allele": ["HLA-A*02:01", "HLA-B*07:02", "HLA-A*02:01"],
                "locus": ["A", "B", "A"],
            }
        )

        # Mock the global lookup_db
        with patch("immunotype.immunotype.lookup_db", mock_lookup_db):
            result = predict_lookup(sample_peptides, sample_alleles)

        assert isinstance(result, pd.DataFrame)
        assert "allele" in result.columns
        assert "probability" in result.columns
        assert "sample" in result.columns

    def test_predict_with_lookup_only(
        self, sample_peptides, sample_alleles, monkeypatch
    ):
        """Test prediction using only lookup method."""
        # Mock lookup database
        mock_lookup_db = pd.DataFrame(
            {
                "peptide": ["ALDGRETD", "ASDSGKYL", "AVDPTSGQ"],
                "allele": ["HLA-A*02:01", "HLA-B*07:02", "HLA-A*02:01"],
                "locus": ["A", "B", "A"],
            }
        )

        with patch("immunotype.immunotype.lookup_db", mock_lookup_db):
            pred_df, typing = predict(
                sample_peptides, sample_alleles, use_gnn=False, use_lookup=True
            )

        assert isinstance(pred_df, pd.DataFrame)
        assert isinstance(typing, pd.DataFrame)
        assert len(pred_df) > 0
        assert "allele" in pred_df.columns
        assert "probability" in pred_df.columns

    def test_predict_invalid_options(self, sample_peptides, sample_alleles):
        """Test that prediction fails when both methods are disabled."""
        with pytest.raises(ValueError, match="Must use GNN or lookup or both"):
            predict(sample_peptides, sample_alleles, use_gnn=False, use_lookup=False)

    def test_prepare_data_lookup_only(self, monkeypatch):
        """Test data preparation for lookup only."""
        mock_lookup_db = pd.DataFrame(
            {"peptide": ["ALDGRETD"], "allele": ["HLA-A*02:01"], "locus": ["A"]}
        )

        mock_read_csv = MagicMock(return_value=mock_lookup_db)
        monkeypatch.setattr("pandas.read_csv", mock_read_csv)

        prepare_data(use_gnn=False, use_lookup=True)

        # Should not raise any errors
        assert True

    def test_predict_empty_peptide_list(self, sample_alleles):
        """Test prediction with empty peptide list."""
        empty_peptides = pd.DataFrame(columns=["peptide", "sample"])

        # Mock lookup database
        mock_lookup_db = pd.DataFrame(
            {"peptide": ["ALDGRETD"], "allele": ["HLA-A*02:01"], "locus": ["A"]}
        )

        with patch("immunotype.immunotype.lookup_db", mock_lookup_db):
            pred_df, typing = predict(
                empty_peptides, sample_alleles, use_gnn=False, use_lookup=True
            )

        assert isinstance(pred_df, pd.DataFrame)
        assert isinstance(typing, pd.DataFrame)

    def test_predict_with_different_sample_ids(self, sample_alleles, monkeypatch):
        """Test prediction with multiple sample IDs."""
        multi_sample_peptides = pd.DataFrame(
            {
                "peptide": ["ALDGRETD", "ASDSGKYL", "AVDPTSGQ"],
                "sample": ["sample1", "sample2", "sample1"],
            }
        )

        mock_lookup_db = pd.DataFrame(
            {
                "peptide": ["ALDGRETD", "ASDSGKYL", "AVDPTSGQ"],
                "allele": ["HLA-A*02:01", "HLA-B*07:02", "HLA-A*02:01"],
                "locus": ["A", "B", "A"],
            }
        )

        with patch("immunotype.immunotype.lookup_db", mock_lookup_db):
            pred_df, typing = predict(
                multi_sample_peptides, sample_alleles, use_gnn=False, use_lookup=True
            )

        assert isinstance(pred_df, pd.DataFrame)
        assert isinstance(typing, pd.DataFrame)
        # Should have results for both samples
        samples_in_result = pred_df["sample"].unique()
        assert len(samples_in_result) > 0
