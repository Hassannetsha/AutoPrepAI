"""Tests for DataTypeInconsistencyAgent."""

import pandas as pd
from unittest.mock import MagicMock, patch

from business_logic.cleaning_coordinator.data_context import DataContext
from business_logic.cleaning_coordinator.agent_params import AgentParams


class TestDataTypeInconsistencyAgent:
    @patch("agents.data_type_inconsistency_agent.DataTypeInconsistencyDetector")
    @patch("agents.data_type_inconsistency_agent.DataResolvingService")
    def test_detects_and_resolves(self, MockResolver, MockDetector):
        from ml_layer.agents.data_type_inconsistency_agent import DataTypeInconsistencyAgent

        mock_detector = MagicMock()
        mock_detector.analyze_dataframe.return_value = {
            "mixed_col": {
                "detected_types": {"integer": 3, "string": 2},
                "recommended_type": "integer"
            }
        }
        MockDetector.return_value = mock_detector

        mock_resolver = MagicMock()
        mock_resolver.df_resolved = pd.DataFrame({"mixed_col": [1, 2, 3, 4, 5]})
        mock_resolver.resolution_log = {"mixed_col": "converted to integer"}
        MockResolver.return_value = mock_resolver

        agent = DataTypeInconsistencyAgent()
        df = pd.DataFrame({"mixed_col": [1, "two", 3, "four", 5]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=[])

        result = agent.execute(ctx, params)
        assert result.metadata["datatype_inconsistencies_fixed"] is True
        assert "datatype_detection_results" in result.metadata

    @patch("agents.data_type_inconsistency_agent.DataTypeInconsistencyDetector")
    def test_no_inconsistencies_returns_early(self, MockDetector):
        from ml_layer.agents.data_type_inconsistency_agent import DataTypeInconsistencyAgent

        mock_detector = MagicMock()
        mock_detector.analyze_dataframe.return_value = {
            "col1": {"detected_types": {"integer": 5}, "recommended_type": "integer"}
        }
        MockDetector.return_value = mock_detector

        agent = DataTypeInconsistencyAgent()
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=[])

        result = agent.execute(ctx, params)
        assert result.metadata["datatype_inconsistencies_fixed"] is True
        assert "resolution_log" not in result.metadata

    @patch("agents.data_type_inconsistency_agent.DataTypeInconsistencyDetector")
    def test_skips_empty_column_recommendation(self, MockDetector):
        from ml_layer.agents.data_type_inconsistency_agent import DataTypeInconsistencyAgent

        mock_detector = MagicMock()
        mock_detector.analyze_dataframe.return_value = {
            "col1": {
                "detected_types": {"integer": 2, "empty": 3},
                "recommended_type": "empty_column"
            }
        }
        MockDetector.return_value = mock_detector

        agent = DataTypeInconsistencyAgent()
        df = pd.DataFrame({"col1": [1, None, None, None, None]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=[])

        with patch(
            "agents.data_type_inconsistency_agent.DataResolvingService"
        ) as MockResolver:
            mock_resolver = MagicMock()
            mock_resolver.df_resolved = df
            MockResolver.return_value = mock_resolver

            result = agent.execute(ctx, params)
            mock_resolver.resolve.assert_not_called()
