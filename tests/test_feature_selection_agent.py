"""Tests for FeatureSelectionAgent."""

import pandas as pd
from unittest.mock import MagicMock, patch

from business_logic.cleaning_coordinator.data_context import DataContext
from business_logic.cleaning_coordinator.agent_params import AgentParams


class TestFeatureSelectionAgent:
    @patch("agents.feature_selection_agent.FeatureSelectionService")
    def test_selects_features(self, MockService):
        from ml_layer.agents.feature_selection_agent import FeatureSelectionAgent

        mock_service = MagicMock()
        df_input = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        df_pruned = pd.DataFrame({"a": [1, 2], "c": [5, 6]})
        mock_service.run.return_value = (["a", "c"], df_pruned)
        MockService.return_value = mock_service

        agent = FeatureSelectionAgent()
        ctx = DataContext(data=df_input)
        params = AgentParams(columns=["a", "b", "c"])

        result = agent.execute(ctx, params)
        assert result.metadata.get("features_selected") is True
        assert result.metadata.get("selected_features") == ["a", "c"]

    @patch("agents.feature_selection_agent.FeatureSelectionService")
    def test_handles_value_error(self, MockService):
        from ml_layer.agents.feature_selection_agent import FeatureSelectionAgent

        mock_service = MagicMock()
        mock_service.run.side_effect = ValueError("Insufficient features")
        MockService.return_value = mock_service

        agent = FeatureSelectionAgent()
        ctx = DataContext(data=pd.DataFrame({"a": [1, 2]}))
        params = AgentParams(columns=[])

        result = agent.execute(ctx, params)
        assert "features_selected" not in result.metadata

    @patch("agents.feature_selection_agent.FeatureSelectionService")
    def test_handles_generic_exception(self, MockService):
        from ml_layer.agents.feature_selection_agent import FeatureSelectionAgent

        mock_service = MagicMock()
        mock_service.run.side_effect = Exception("Unexpected error")
        MockService.return_value = mock_service

        agent = FeatureSelectionAgent()
        ctx = DataContext(data=pd.DataFrame({"a": [1, 2]}))
        params = AgentParams(columns=[])

        result = agent.execute(ctx, params)
        assert "features_selected" not in result.metadata

    @patch("agents.feature_selection_agent.FeatureSelectionService")
    def test_passes_threshold_and_n_features(self, MockService):
        from ml_layer.agents.feature_selection_agent import FeatureSelectionAgent

        mock_service = MagicMock()
        mock_service.run.return_value = (["a"], pd.DataFrame({"a": [1, 2]}))
        MockService.return_value = mock_service

        agent = FeatureSelectionAgent()
        ctx = DataContext(data=pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
        params = AgentParams(
            columns=["a", "b"],
            options={"threshold": 0.1, "n_features": 5}
        )

        result = agent.execute(ctx, params)
        mock_service.run.assert_called_once()
        _, kwargs_call = mock_service.run.call_args
        assert kwargs_call["threshold"] == 0.1
        assert kwargs_call["n_features"] == 5
