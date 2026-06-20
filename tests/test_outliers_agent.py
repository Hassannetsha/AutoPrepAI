"""Tests for OutliersAgent."""

import pandas as pd
from unittest.mock import MagicMock, patch

from data_context import DataContext
from agent_params import AgentParams


class TestOutliersAgent:
    @patch("agents.outliers_agent.OutliersService")
    def test_execute_processes_data(self, MockOutliersService):
        from agents.outliers_agent import OutliersAgent

        mock_service = MagicMock()
        df_input = pd.DataFrame({"x": [1, 2, 3]})
        df_output = pd.DataFrame({"x": [1, 2]})
        mock_service.process.return_value = df_output
        MockOutliersService.return_value = mock_service

        agent = OutliersAgent()
        ctx = DataContext(data=df_input)
        params = AgentParams()

        result = agent.execute(ctx, params)
        assert result.metadata.get("outliers_handled") is True
        assert len(result.data) == 2
        MockOutliersService.assert_called_once_with(dataframe=df_input)

    @patch("agents.outliers_agent.OutliersService")
    def test_execute_passes_params_ignored(self, MockOutliersService):
        from agents.outliers_agent import OutliersAgent

        mock_service = MagicMock()
        mock_service.process.return_value = pd.DataFrame({"x": [1]})
        MockOutliersService.return_value = mock_service

        agent = OutliersAgent()
        ctx = DataContext(data=pd.DataFrame({"x": [1, 2, 3, 100]}))
        params = AgentParams(columns=["x"], strategy="iqr")

        result = agent.execute(ctx, params)
        assert result.metadata.get("outliers_handled") is True
