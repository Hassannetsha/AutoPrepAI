"""Tests for ScalingAgent."""

import pandas as pd
from unittest.mock import MagicMock, patch

from business_logic.cleaning_coordinator.data_context import DataContext
from business_logic.cleaning_coordinator.agent_params import AgentParams


class TestScalingAgent:
    @patch("agents.scaling_agent.DFScaler")
    def test_execute_standard_scale(self, MockDFScaler):
        from ml_layer.agents.scaling_agent import ScalingAgent

        mock_scaler = MagicMock()
        df_scaled = pd.DataFrame({"age": [-1.0, 1.0]})
        mock_scaler.scale.return_value = df_scaled
        MockDFScaler.return_value = mock_scaler

        agent = ScalingAgent()
        df = pd.DataFrame({"age": [25.0, 75.0], "name": ["a", "b"]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=[], strategy="standard")

        result = agent.execute(ctx, params)
        assert result.metadata.get("scaled") is True
        assert "scaling_fit" in result.metadata
        assert result.metadata["scaling_fit"]["method"] == "standard"

    @patch("agents.scaling_agent.DFScaler")
    def test_execute_minmax_scale(self, MockDFScaler):
        from ml_layer.agents.scaling_agent import ScalingAgent

        mock_scaler = MagicMock()
        df_scaled = pd.DataFrame({"age": [0.0, 1.0]})
        mock_scaler.scale.return_value = df_scaled
        MockDFScaler.return_value = mock_scaler

        agent = ScalingAgent()
        df = pd.DataFrame({"age": [10.0, 20.0]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=[], strategy="minmax")

        result = agent.execute(ctx, params)
        assert result.metadata["scaling_fit"]["method"] == "minmax"

    @patch("agents.scaling_agent.DFScaler")
    def test_execute_method_from_columns(self, MockDFScaler):
        from ml_layer.agents.scaling_agent import ScalingAgent

        mock_scaler = MagicMock()
        mock_scaler.scale.return_value = pd.DataFrame({"age": [0.0, 1.0]})
        MockDFScaler.return_value = mock_scaler

        agent = ScalingAgent()
        df = pd.DataFrame({"age": [10.0, 20.0]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=[["age"], "minmax"], strategy="standard")

        result = agent.execute(ctx, params)
        assert result.metadata["scaling_fit"]["method"] == "minmax"

    @patch("agents.scaling_agent.DFScaler")
    def test_execute_excludes_target_col(self, MockDFScaler):
        from ml_layer.agents.scaling_agent import ScalingAgent

        mock_scaler = MagicMock()
        mock_scaler.scale.return_value = pd.DataFrame({"age": [-1.0, 1.0], "price": [100, 200]})
        MockDFScaler.return_value = mock_scaler

        agent = ScalingAgent()
        df = pd.DataFrame({"age": [25.0, 75.0], "price": [100, 200]})
        ctx = DataContext(data=df, metadata={"target_col": "price"})
        params = AgentParams(columns=[], strategy="standard")

        result = agent.execute(ctx, params)
        scaled_cols = result.metadata["scaling_fit"]["columns"]
        assert "price" not in scaled_cols

    @patch("agents.scaling_agent.DFScaler")
    def test_execute_with_comma_separated_columns(self, MockDFScaler):
        from ml_layer.agents.scaling_agent import ScalingAgent

        mock_scaler = MagicMock()
        mock_scaler.scale.return_value = pd.DataFrame({"x": [0.0], "y": [0.0]})
        MockDFScaler.return_value = mock_scaler

        agent = ScalingAgent()
        df = pd.DataFrame({"x": [1.0], "y": [2.0], "z": ["a"]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=["x, y"], strategy="minmax")

        result = agent.execute(ctx, params)
        assert result.metadata.get("scaled") is True
