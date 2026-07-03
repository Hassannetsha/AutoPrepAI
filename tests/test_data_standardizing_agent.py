"""Tests for DataStandardizerAgent."""

import pandas as pd
from unittest.mock import MagicMock, patch

from business_logic.cleaning_coordinator.data_context import DataContext
from business_logic.cleaning_coordinator.agent_params import AgentParams


# Patch targets for dependencies
_PATCHES = {
    "groq": "ml_layer.agents.data_standardizing_agent.Groq",
    "service": "ml_layer.agents.data_standardizing_agent.DataStandardizingService",
    "key_manager": "ml_layer.agents.data_standardizing_agent.get_key_manager",
    "rate_limiter": "ml_layer.agents.data_standardizing_agent.RateLimiter",
    "llm_client": "ml_layer.agents.data_standardizing_agent.GroqLLMClient",
}


def _make_mock_service(df: pd.DataFrame) -> MagicMock:
    """Build a mock DataStandardizingService with the correct results shape."""
    mock_service = MagicMock()
    mock_service.df = df
    mock_service.results = {
        # ❌ REMOVED: "numeric_issues": {} (no longer handled by this service)
        "standardization": {},
        "validation_log": [],
    }
    return mock_service


def _make_mock_key_manager() -> MagicMock:
    mock = MagicMock()
    mock.get_current_key.return_value = "fake_key"
    return mock


class TestDataStandardizerAgent:

    @patch(_PATCHES["llm_client"])
    @patch(_PATCHES["rate_limiter"])
    @patch(_PATCHES["key_manager"])
    @patch(_PATCHES["service"])
    @patch(_PATCHES["groq"])
    def test_execute_standardize(
        self, MockGroq, MockService, MockKeyManager, MockRateLimiter, MockLLMClient
    ):
        from ml_layer.agents.data_standardizing_agent import DataStandardizerAgent

        MockKeyManager.return_value = _make_mock_key_manager()
        MockService.return_value = _make_mock_service(
            pd.DataFrame({"color": ["red", "blue"]})
        )

        agent = DataStandardizerAgent()
        df = pd.DataFrame({"color": ["red", "BLUE"], "val": [1, 2]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=["color"])

        result = agent.execute(ctx, params)
        assert result.metadata.get("data_standardized") is True
        assert "standardization_results" in result.metadata

    @patch(_PATCHES["llm_client"])
    @patch(_PATCHES["rate_limiter"])
    @patch(_PATCHES["key_manager"])
    @patch(_PATCHES["service"])
    @patch(_PATCHES["groq"])
    def test_skips_when_requested_columns_missing(
        self, MockGroq, MockService, MockKeyManager, MockRateLimiter, MockLLMClient
    ):
        from ml_layer.agents.data_standardizing_agent import DataStandardizerAgent

        MockKeyManager.return_value = _make_mock_key_manager()

        agent = DataStandardizerAgent()
        df = pd.DataFrame({"a": [1, 2]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=["nonexistent"])

        result = agent.execute(ctx, params)
        assert result.metadata.get("data_standardized") is False
        MockService.assert_not_called()

    @patch(_PATCHES["llm_client"])
    @patch(_PATCHES["rate_limiter"])
    @patch(_PATCHES["key_manager"])
    @patch(_PATCHES["service"])
    @patch(_PATCHES["groq"])
    def test_skips_when_no_categorical_columns(
        self, MockGroq, MockService, MockKeyManager, MockRateLimiter, MockLLMClient
    ):
        """Service should not be constructed if only numeric columns are present."""
        from ml_layer.agents.data_standardizing_agent import DataStandardizerAgent

        MockKeyManager.return_value = _make_mock_key_manager()

        agent = DataStandardizerAgent()
        # "val" is numeric, so it gets ignored. Empty list = skip.
        df = pd.DataFrame({"val": [1, 2]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=[])

        result = agent.execute(ctx, params)
        assert result.metadata.get("data_standardized") is False
        MockService.assert_not_called()

    @patch(_PATCHES["llm_client"])
    @patch(_PATCHES["rate_limiter"])
    @patch(_PATCHES["key_manager"])
    @patch(_PATCHES["service"])
    @patch(_PATCHES["groq"])
    def test_builds_validation_layer_from_params(
        self, MockGroq, MockService, MockKeyManager, MockRateLimiter, MockLLMClient
    ):
        from ml_layer.agents.data_standardizing_agent import DataStandardizerAgent

        MockKeyManager.return_value = _make_mock_key_manager()
        MockService.return_value = _make_mock_service(
            pd.DataFrame({"color": ["red"]})
        )

        agent = DataStandardizerAgent()
        df = pd.DataFrame({"color": ["red"], "val": [1]})
        ctx = DataContext(data=df)
        params = AgentParams(
            columns=["color"],
            options={
                "validation_rules": {"color": {"allowed_values": {"red", "blue"}}}
            },
        )

        result = agent.execute(ctx, params)
        assert result.metadata.get("data_standardized") is True

    @patch(_PATCHES["llm_client"])
    @patch(_PATCHES["rate_limiter"])
    @patch(_PATCHES["key_manager"])
    @patch(_PATCHES["service"])
    @patch(_PATCHES["groq"])
    def test_rate_limiter_and_llm_client_constructed_with_params(
        self, MockGroq, MockService, MockKeyManager, MockRateLimiter, MockLLMClient
    ):
        """RateLimiter and GroqLLMClient receive the right params from AgentParams."""
        from ml_layer.agents.data_standardizing_agent import DataStandardizerAgent

        MockKeyManager.return_value = _make_mock_key_manager()
        MockService.return_value = _make_mock_service(
            pd.DataFrame({"color": ["red"]})
        )

        agent = DataStandardizerAgent()
        df = pd.DataFrame({"color": ["red"], "val": [1]})
        ctx = DataContext(data=df)
        params = AgentParams(
            columns=["color"],
            options={"requests_per_minute": 10, "tokens_per_minute": 15_000},
        )

        agent.execute(ctx, params)

        MockRateLimiter.assert_called_once_with(
            requests_per_minute=10,
            tokens_per_minute=15_000,
        )
        MockLLMClient.assert_called_once()