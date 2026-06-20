"""Tests for SpellingCorrectorAgent."""

import pandas as pd
from unittest.mock import MagicMock, patch

from data_context import DataContext
from agent_params import AgentParams


class TestSpellingCorrectorAgent:
    @patch("agents.spelling_corrector_agent.SpellingCorrectorService")
    def test_corrects_spelling(self, MockService):
        from agents.spelling_corrector_agent import SpellingCorrectorAgent

        mock_service = MagicMock()
        df_input = pd.DataFrame({"color": ["red", "blu", "red", "gren"]})
        df_corrected = pd.DataFrame({"color": ["red", "blue", "red", "green"]})
        mock_service.correct_dataframe_column.return_value = df_corrected["color"]
        MockService.return_value = mock_service

        agent = SpellingCorrectorAgent()
        ctx = DataContext(data=df_input)
        params = AgentParams(columns=["color"])

        result = agent.execute(ctx, params)
        assert result.metadata["spelling_corrected"] is True
        assert "color" in result.metadata["spelling_corrected_columns"]
        mock_service.build_dictionary_from_dataframe.assert_called()
        mock_service.correct_dataframe_column.assert_called()

    @patch("agents.spelling_corrector_agent.SpellingCorrectorService")
    def test_no_categorical_columns(self, MockService):
        from agents.spelling_corrector_agent import SpellingCorrectorAgent

        agent = SpellingCorrectorAgent()
        df = pd.DataFrame({"x": [1, 2, 3]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=[])

        result = agent.execute(ctx, params)
        MockService.assert_not_called()
        assert "spelling_corrected" not in result.metadata

    @patch("agents.spelling_corrector_agent.SpellingCorrectorService")
    def test_specified_columns_not_categorical(self, MockService):
        from agents.spelling_corrector_agent import SpellingCorrectorAgent

        agent = SpellingCorrectorAgent()
        df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=["x"])

        result = agent.execute(ctx, params)
        MockService.assert_not_called()

    @patch("agents.spelling_corrector_agent.SpellingCorrectorService")
    def test_corrects_all_categorical_when_no_columns_specified(self, MockService):
        from agents.spelling_corrector_agent import SpellingCorrectorAgent

        mock_service = MagicMock()
        mock_service.correct_dataframe_column.return_value = pd.Series(["a", "b"])
        MockService.return_value = mock_service

        agent = SpellingCorrectorAgent()
        df = pd.DataFrame({"cat1": ["x", "y"], "cat2": ["p", "q"]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=[])

        result = agent.execute(ctx, params)
        assert len(result.metadata["spelling_corrected_columns"]) == 2

    @patch("agents.spelling_corrector_agent.SpellingCorrectorService")
    def test_error_on_one_column_continues(self, MockService):
        from agents.spelling_corrector_agent import SpellingCorrectorAgent

        mock_service = MagicMock()
        mock_service.build_dictionary_from_dataframe.side_effect = [
            None, Exception("fail")
        ]
        mock_service.correct_dataframe_column.side_effect = [
            pd.Series(["a", "b"]), None
        ]
        MockService.return_value = mock_service

        agent = SpellingCorrectorAgent()
        df = pd.DataFrame({"cat1": ["x", "y"], "cat2": ["p", "q"]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=[])

        result = agent.execute(ctx, params)
        assert len(result.metadata["spelling_corrected_columns"]) == 1
