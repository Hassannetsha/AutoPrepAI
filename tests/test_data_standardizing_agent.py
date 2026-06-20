"""Tests for DataStandardizerAgent."""

import pandas as pd
from unittest.mock import MagicMock, patch

from data_context import DataContext
from agent_params import AgentParams


class TestDataStandardizerAgent:
    @patch("agents.data_standardizing_agent.Groq")
    @patch("agents.data_standardizing_agent.DataStandardizingService")
    @patch("agents.data_standardizing_agent.get_key_manager")
    def test_execute_standardize(self, MockKeyManager, MockService, MockGroq):
        from agents.data_standardizing_agent import DataStandardizerAgent

        mock_key_mgr = MagicMock()
        mock_key_mgr.get_current_key.return_value = "fake_key"
        MockKeyManager.return_value = mock_key_mgr

        mock_service = MagicMock()
        mock_service.df = pd.DataFrame({"color": ["red", "BLUE"]})
        mock_service.results = {"color": {"red": "red", "BLUE": "blue"}}
        MockService.return_value = mock_service

        agent = DataStandardizerAgent()
        df = pd.DataFrame({"color": ["red", "BLUE"], "val": [1, 2]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=["color"])

        result = agent.execute(ctx, params)
        assert result.metadata.get("data_standardized") is True
        assert "standardization_results" in result.metadata

    @patch("agents.data_standardizing_agent.Groq")
    @patch("agents.data_standardizing_agent.DataStandardizingService")
    @patch("agents.data_standardizing_agent.get_key_manager")
    def test_skips_when_requested_columns_missing(
        self, MockKeyManager, MockService, MockGroq
    ):
        from agents.data_standardizing_agent import DataStandardizerAgent

        mock_key_mgr = MagicMock()
        mock_key_mgr.get_current_key.return_value = "fake_key"
        MockKeyManager.return_value = mock_key_mgr

        agent = DataStandardizerAgent()
        df = pd.DataFrame({"a": [1, 2]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=["nonexistent"])

        result = agent.execute(ctx, params)
        assert result.metadata.get("data_standardized") is False

    @patch("agents.data_standardizing_agent.Groq")
    @patch("agents.data_standardizing_agent.DataStandardizingService")
    @patch("agents.data_standardizing_agent.get_key_manager")
    def test_skips_when_no_categorical_or_numeric_columns(
        self, MockKeyManager, MockService, MockGroq
    ):
        from agents.data_standardizing_agent import DataStandardizerAgent

        mock_key_mgr = MagicMock()
        mock_key_mgr.get_current_key.return_value = "fake_key"
        MockKeyManager.return_value = mock_key_mgr

        agent = DataStandardizerAgent()
        df = pd.DataFrame({"a": [1, 2]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=[])

        result = agent.execute(ctx, params)
        assert result.metadata.get("data_standardized") is False

    @patch("agents.data_standardizing_agent.Groq")
    @patch("agents.data_standardizing_agent.DataStandardizingService")
    @patch("agents.data_standardizing_agent.get_key_manager")
    def test_builds_validation_layer_from_params(
        self, MockKeyManager, MockService, MockGroq
    ):
        from agents.data_standardizing_agent import DataStandardizerAgent

        mock_key_mgr = MagicMock()
        mock_key_mgr.get_current_key.return_value = "fake_key"
        MockKeyManager.return_value = mock_key_mgr

        mock_service = MagicMock()
        mock_service.df = pd.DataFrame({"color": ["red"]})
        mock_service.results = {}
        MockService.return_value = mock_service

        agent = DataStandardizerAgent()
        df = pd.DataFrame({"color": ["red"], "val": [1]})
        ctx = DataContext(data=df)
        params = AgentParams(
            columns=["color"],
            options={
                "validation_rules": {"color": {"allowed_values": {"red", "blue"}}}
            }
        )

        result = agent.execute(ctx, params)
        assert result.metadata.get("data_standardized") is True
