"""
Tests for the core immunotype functionality.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from immunotype.immunotype import predict

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestImmunotype:
    """Test cases for core immunotype functionality."""

    # DRY: Reusable mock data
    mock_lookup_db = pd.DataFrame({
        "peptide": ["ALDGRETD", "ASDSGKYL", "AVDPTSGQ"],
        "allele": ["HLA-A*02:01", "HLA-B*07:02", "HLA-A*02:01"],
        "locus": ["A", "B", "A"],
    })

    @staticmethod
    def setup_gnn_mocks():
        """Setup common GNN mocks for tests that use GNN prediction."""
        mock_model = MagicMock()
        mock_gnn_class = MagicMock(return_value=mock_model)
        mock_load_weights = MagicMock(return_value=mock_model)
        
        # Mock data and dataloader
        mock_data = MagicMock()
        mock_get_data = MagicMock(return_value=mock_data)
        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_dataloader = MagicMock(return_value=[mock_batch])
        
        # Mock model evaluation
        mock_model.eval.return_value = None
        
        return {
            'gnn_class': mock_gnn_class,
            'load_weights': mock_load_weights,
            'get_data': mock_get_data,
            'dataloader': mock_dataloader,
            'model': mock_model,
            'batch': mock_batch
        }

    def test_predict_lookup(
        self, sample_peptides, sample_alleles, monkeypatch
    ):
        """Test prediction using only lookup method."""
        with patch("immunotype.immunotype.LOOKUP_DF", self.mock_lookup_db):
            pred_df, typing = predict(
                sample_peptides, sample_alleles, prediction_model="lookup"
            )

        assert len(pred_df) > 0
        assert "allele" in pred_df.columns
        assert "probability" in pred_df.columns
        assert "sample" in pred_df.columns
        assert "locus" in pred_df.columns


    def test_predict_empty_peptide_list(self, sample_alleles):
        """Test prediction with empty peptide list."""
        empty_peptides = pd.DataFrame(columns=["peptide", "sample"])

        with patch("immunotype.immunotype.LOOKUP_DF", self.mock_lookup_db):
            pred_df, typing = predict(
                empty_peptides, sample_alleles, prediction_model="lookup"
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

        with patch("immunotype.immunotype.LOOKUP_DF", self.mock_lookup_db):
            pred_df, typing = predict(
                multi_sample_peptides, sample_alleles, prediction_model="lookup"
            )

        assert isinstance(pred_df, pd.DataFrame)
        assert isinstance(typing, pd.DataFrame)
        # Should have results for both samples
        samples_in_result = pred_df["sample"].unique()
        assert len(samples_in_result) > 0

    def test_predict_with_gnn_model(self, sample_peptides, sample_alleles):
        """Test prediction using GNN model."""
        mocks = self.setup_gnn_mocks()
        
        # Setup specific mock data for GNN test
        mocks['batch'].sample = ["sample1", "sample2"]
        # Mock model output - needs to match shape (len(batch.sample), num_alleles)
        # 2 samples, 3 alleles = 6 values total
        mocks['model'].return_value.cpu.return_value.detach.return_value.numpy.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
        
        with patch("immunotype.immunotype.GNN", mocks['gnn_class']), \
             patch("immunotype.immunotype.load_weights", mocks['load_weights']), \
             patch("immunotype.immunotype.get_hetero_data", mocks['get_data']), \
             patch("immunotype.immunotype.DataLoader", mocks['dataloader']), \
             patch("torch.cuda.is_available", return_value=False):
            
            pred_df, typing_df = predict(
                sample_peptides, sample_alleles, prediction_model="gnn", device="cpu"
            )

        assert "probability" in pred_df.columns

    def test_predict_with_ensemble_model(self, sample_peptides, sample_alleles):
        """Test prediction using ensemble model."""
        mocks = self.setup_gnn_mocks()
        
        # Setup specific mock data for ensemble test
        mocks['batch'].sample = ["0"]
        mocks['model'].return_value.cpu.return_value.detach.return_value.numpy.return_value = [[0.1, 0.2, 0.3]]
        
        # Mock everything needed for both GNN and lookup
        with patch("immunotype.immunotype.LOOKUP_DF", self.mock_lookup_db), \
             patch("immunotype.immunotype.GNN", mocks['gnn_class']), \
             patch("immunotype.immunotype.load_weights", mocks['load_weights']), \
             patch("immunotype.immunotype.get_hetero_data", mocks['get_data']), \
             patch("immunotype.immunotype.DataLoader", mocks['dataloader']), \
             patch("torch.cuda.is_available", return_value=False):
            
            pred_df, typing_df = predict(
                sample_peptides, sample_alleles, prediction_model="ensemble", device="cpu"
            )

        assert "probability" in pred_df.columns
        assert "probability_gnn" in pred_df.columns
        assert "probability_lookup" in pred_df.columns