"""Tests for NLPAgent."""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from data_context import DataContext
from agent_params import AgentParams


class TestNLPAgent:
    @patch("agents.nlp_agent.NLPService")
    def test_execute_returns_intents(self, MockNLPService):
        from agents.nlp_agent import NLPAgent

        mock_service = MagicMock()
        mock_service.run.return_value = (
            pd.DataFrame({"x": [1, 2]}),
            [["handle_missing_values", "x"]]
        )
        MockNLPService.return_value = mock_service
        NLPAgent._nlp_service = mock_service

        agent = NLPAgent()
        ctx = DataContext(data=pd.DataFrame({"x": [1, 2]}))
        params = AgentParams(options={"user_command": "handle missing values"})

        result = agent.execute(ctx, params)
        assert result.metadata.get("nlp_done") is True
        assert result.metadata.get("intents") == [["handle_missing_values", "x"]]

    @patch("agents.nlp_agent.NLPService")
    def test_execute_skips_when_nlp_done(self, MockNLPService):
        from agents.nlp_agent import NLPAgent

        mock_service = MagicMock()
        MockNLPService.return_value = mock_service
        NLPAgent._nlp_service = mock_service

        agent = NLPAgent()
        ctx = DataContext(
            data=pd.DataFrame({"x": [1]}),
            metadata={"nlp_done": True}
        )
        params = AgentParams()

        result = agent.execute(ctx, params)
        mock_service.run.assert_not_called()

    @patch("agents.nlp_agent.NLPService")
    def test_execute_tuple_with_intents_only(self, MockNLPService):
        from agents.nlp_agent import NLPAgent

        mock_service = MagicMock()
        mock_service.run.return_value = (["remove_outliers"],)
        MockNLPService.return_value = mock_service
        NLPAgent._nlp_service = mock_service

        agent = NLPAgent()
        ctx = DataContext(data=pd.DataFrame({"x": [1]}))
        params = AgentParams(options={"user_command": "test"})

        result = agent.execute(ctx, params)
        assert result.metadata["intents"] == ["remove_outliers"]

    @patch("agents.nlp_agent.NLPService")
    def test_execute_list_result(self, MockNLPService):
        from agents.nlp_agent import NLPAgent

        mock_service = MagicMock()
        mock_service.run.return_value = [["scale_numerical"]]
        MockNLPService.return_value = mock_service
        NLPAgent._nlp_service = mock_service

        agent = NLPAgent()
        ctx = DataContext(data=pd.DataFrame({"x": [1]}))
        params = AgentParams(options={"user_command": "scale"})

        result = agent.execute(ctx, params)
        assert result.metadata["intents"] == [["scale_numerical"]]

    @patch("agents.nlp_agent.NLPService")
    def test_execute_handles_error(self, MockNLPService):
        from agents.nlp_agent import NLPAgent

        mock_service = MagicMock()
        mock_service.run.side_effect = Exception("test error")
        MockNLPService.return_value = mock_service
        NLPAgent._nlp_service = mock_service

        agent = NLPAgent()
        ctx = DataContext(data=pd.DataFrame({"x": [1]}))
        params = AgentParams(options={"user_command": "test"})

        result = agent.execute(ctx, params)
        # On non-RuntimeError exceptions, agent returns without setting nlp_done
        assert "nlp_done" not in result.metadata

    @patch("agents.nlp_agent.NLPService")
    def test_execute_re_raises_runtime_error(self, MockNLPService):
        from agents.nlp_agent import NLPAgent

        mock_service = MagicMock()
        mock_service.run.side_effect = RuntimeError("API key exhausted")
        MockNLPService.return_value = mock_service
        NLPAgent._nlp_service = mock_service

        agent = NLPAgent()
        ctx = DataContext(data=pd.DataFrame({"x": [1]}))
        params = AgentParams(options={"user_command": "test"})

        with pytest.raises(RuntimeError, match="API key exhausted"):
            agent.execute(ctx, params)

    @patch("agents.nlp_agent.NLPService")
    def test_user_command_from_context_fallback(self, MockNLPService):
        from agents.nlp_agent import NLPAgent

        mock_service = MagicMock()
        mock_service.run.return_value = (pd.DataFrame({"x": [1]}), [])
        MockNLPService.return_value = mock_service
        NLPAgent._nlp_service = mock_service

        agent = NLPAgent()
        ctx = DataContext(
            data=pd.DataFrame({"x": [1]}),
            metadata={"user_command": "fallback command"}
        )
        params = AgentParams()

        result = agent.execute(ctx, params)
        mock_service.run.assert_called_once()
        args, kwargs = mock_service.run.call_args
        assert kwargs["user_input"] == "fallback command"
