"""Tests for FeatureEngineeringAgent."""

import pandas as pd
from unittest.mock import MagicMock, patch, ANY

from data_context import DataContext
from agent_params import AgentParams


class TestFeatureEngineeringAgent:
    @patch("agents.feature_engineering_agent.dspy.ChainOfThought")
    @patch("agents.feature_engineering_agent.FeatureEngineeringService")
    @patch("agents.feature_engineering_agent.get_key_manager")
    def test_execute_generates_features(
        self, MockKeyManager, MockFEService, MockChainOfThought
    ):
        from agents.feature_engineering_agent import FeatureEngineeringAgent

        # Mock the key manager
        mock_key_mgr = MagicMock()
        mock_key_mgr.get_current_key.return_value = "fake_key"
        MockKeyManager.return_value = mock_key_mgr

        # Mock the DSPy predictor
        mock_predictor = MagicMock()
        mock_result = MagicMock()
        mock_result.suggested_features = "age_squared: Square of age"
        mock_predictor.return_value = mock_result
        MockChainOfThought.return_value = mock_predictor

        # Mock the feature engineering service
        mock_fe_service = MagicMock()
        new_df = pd.DataFrame({"age": [25, 30], "age_squared": [625, 900]})
        mock_fe_service.engineer.return_value = (new_df, 1)
        MockFEService.return_value = mock_fe_service

        agent = FeatureEngineeringAgent()
        df = pd.DataFrame({"age": [25, 30], "target": [1, 0]})
        ctx = DataContext(data=df, metadata={"target_col": "target"})
        params = AgentParams(options={"top_n": "3"})

        result = agent.execute(ctx, params)
        assert result.metadata.get("features_engineered") is True
        assert result.metadata.get("features_added_count") == 1
        assert result.metadata.get("feature_engineering_suggestions") is not None

    @patch("agents.feature_engineering_agent.dspy.ChainOfThought")
    @patch("agents.feature_engineering_agent.FeatureEngineeringService")
    @patch("agents.feature_engineering_agent.get_key_manager")
    def test_skips_when_no_suggestions(
        self, MockKeyManager, MockFEService, MockChainOfThought
    ):
        from agents.feature_engineering_agent import FeatureEngineeringAgent

        mock_key_mgr = MagicMock()
        mock_key_mgr.get_current_key.return_value = "fake_key"
        MockKeyManager.return_value = mock_key_mgr

        mock_predictor = MagicMock()
        mock_result = MagicMock()
        mock_result.suggested_features = ""
        mock_predictor.return_value = mock_result
        MockChainOfThought.return_value = mock_predictor

        agent = FeatureEngineeringAgent()
        df = pd.DataFrame({"age": [25, 30]})
        ctx = DataContext(data=df)
        params = AgentParams()

        result = agent.execute(ctx, params)
        assert "features_engineered" not in result.metadata
