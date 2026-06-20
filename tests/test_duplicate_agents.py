"""Tests for DuplicateRemoverAgent, ExactDuplicateRemover, SemanticDuplicateRemover."""

import pandas as pd
from unittest.mock import MagicMock, patch

from data_context import DataContext
from agent_params import AgentParams


class TestExactDuplicateRemover:
    @patch("agents.exact_duplicates_agent.ExactDuplicateRemoverService")
    def test_removes_duplicates(self, MockService):
        from agents.exact_duplicates_agent import ExactDuplicateRemover

        mock_service = MagicMock()
        df_input = pd.DataFrame({"id": [1, 1, 2], "val": [10, 10, 20]})
        df_dedup = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        df_dups = pd.DataFrame({"id": [1], "val": [10]})
        mock_service.remove_duplicates.return_value = (df_dedup, df_dups)
        MockService.return_value = mock_service

        agent = ExactDuplicateRemover()
        ctx = DataContext(data=df_input)
        params = AgentParams(columns=["id"])

        result = agent.execute(ctx, params)
        assert result.metadata["exact_duplicates_removed"] is True
        assert result.metadata["exact_duplicates_count"] == 1
        assert len(result.data) == 2

    @patch("agents.exact_duplicates_agent.ExactDuplicateRemoverService")
    def test_no_duplicates_found(self, MockService):
        from agents.exact_duplicates_agent import ExactDuplicateRemover

        mock_service = MagicMock()
        df_input = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
        df_dedup = df_input.copy()
        df_dups = pd.DataFrame()
        mock_service.remove_duplicates.return_value = (df_dedup, df_dups)
        MockService.return_value = mock_service

        agent = ExactDuplicateRemover()
        ctx = DataContext(data=df_input)
        params = AgentParams(columns=["id"])

        result = agent.execute(ctx, params)
        assert result.metadata["exact_duplicates_count"] == 0

    @patch("agents.exact_duplicates_agent.ExactDuplicateRemoverService")
    def test_handles_target_col_in_subset(self, MockService):
        from agents.exact_duplicates_agent import ExactDuplicateRemover

        mock_service = MagicMock()
        df_input = pd.DataFrame(
            {"id": [1, 1], "val": [10, 10], "target": [0, 0]}
        )
        df_dedup = pd.DataFrame(
            {"id": [1], "val": [10], "target": [0]}, index=[0]
        )
        df_dups = pd.DataFrame(
            {"id": [1], "val": [10], "target": [0]}, index=[1]
        )
        mock_service.remove_duplicates.return_value = (df_dedup, df_dups)
        MockService.return_value = mock_service

        agent = ExactDuplicateRemover()
        ctx = DataContext(
            data=df_input,
            metadata={"target_col": "target"}
        )
        params = AgentParams(columns=["id", "val"])

        result = agent.execute(ctx, params)
        MockService.assert_called_once()
        args, kwargs = MockService.call_args
        assert kwargs["subset"] == ["id", "val", "target"]


class TestSemanticDuplicateRemover:
    @patch("agents.semantic_duplicate_remover.SemanticDuplicateRemoverService")
    def test_no_text_columns_skips(self, MockService):
        from agents.semantic_duplicate_remover import SemanticDuplicateRemover

        agent = SemanticDuplicateRemover()
        df = pd.DataFrame({"a": [1, 2]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=[])

        result = agent.execute(ctx, params)
        MockService.assert_not_called()

    @patch("agents.semantic_duplicate_remover.SemanticDuplicateRemoverService")
    def test_short_text_columns_skips(self, MockService):
        from agents.semantic_duplicate_remover import SemanticDuplicateRemover

        agent = SemanticDuplicateRemover()
        df = pd.DataFrame({"label": ["yes", "no", "maybe"]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=[])

        result = agent.execute(ctx, params)
        MockService.assert_not_called()

    @patch("agents.semantic_duplicate_remover.SemanticDuplicateRemoverService")
    def test_semantic_removal(self, MockService):
        from agents.semantic_duplicate_remover import SemanticDuplicateRemover

        mock_service = MagicMock()
        df_input = pd.DataFrame({
            "text": [
                "The quick brown fox jumps over the lazy dog",
                "The quick brown fox jumps over the lazy dog",
                "A completely unique sentence about data science",
            ]
        })
        df_dedup = pd.DataFrame({
            "text": [
                "The quick brown fox jumps over the lazy dog",
                "A completely unique sentence about data science",
            ]
        })
        df_dups = pd.DataFrame({
            "text": ["The quick brown fox jumps over the lazy dog"]
        })
        mock_service.remove_duplicates.return_value = (df_dedup, df_dups)
        MockService.return_value = mock_service

        agent = SemanticDuplicateRemover()
        ctx = DataContext(data=df_input)
        params = AgentParams(columns=["text"])

        result = agent.execute(ctx, params)
        assert result.metadata["semantic_duplicates_removed"] is True
        assert result.metadata["semantic_duplicates_count"] == 1
        assert len(result.data) == 2
        assert result.metadata["semantic_column_used"] == "text"


class TestDuplicateRemoverAgent:
    @patch("agents.duplicate_remover_agent.ExactDuplicateRemover")
    @patch("agents.duplicate_remover_agent.SemanticDuplicateRemover")
    def test_runs_both_sub_agents(self, MockSemantic, MockExact):
        from agents.duplicate_remover_agent import DuplicateRemoverAgent

        mock_exact = MagicMock()
        mock_semantic = MagicMock()
        # Chain: exact_remover.execute(ctx, params) returns the same ctx
        # so semantic_remover receives the same one.
        ctx_after_exact = DataContext(data=pd.DataFrame({"x": [1, 2]}), metadata={})
        mock_exact.execute.return_value = ctx_after_exact
        mock_semantic.execute.return_value = DataContext(
            data=pd.DataFrame({"x": [1, 2]}),
            metadata={"duplicates_removed": True}
        )
        MockExact.return_value = mock_exact
        MockSemantic.return_value = mock_semantic

        agent = DuplicateRemoverAgent()
        df = pd.DataFrame({"x": [1, 1, 2]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=["x"])

        result = agent.execute(ctx, params)
        # metadata starts empty, so "duplicates_removed" from the
        # agent itself is set on the original ctx, not the return value.
        # The result is the return of semantic_remover.
        assert result is mock_semantic.execute.return_value
        mock_exact.execute.assert_called_once()
        mock_semantic.execute.assert_called_once()
