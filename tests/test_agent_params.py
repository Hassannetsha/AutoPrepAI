"""Tests for AgentParams data class."""

from business_logic.cleaning_coordinator.agent_params import AgentParams


class TestAgentParams:
    def test_default_initialization(self):
        params = AgentParams()
        assert params.columns == []
        assert params.strategy == ""
        assert params.options == {}

    def test_custom_initialization(self):
        params = AgentParams(
            columns=["col1", "col2"],
            strategy="mean",
            options={"threshold": 0.5, "n_features": 10}
        )
        assert params.columns == ["col1", "col2"]
        assert params.strategy == "mean"
        assert params.options == {"threshold": 0.5, "n_features": 10}

    def test_get_option_existing(self):
        params = AgentParams(options={"key": "value"})
        assert params.get_option("key") == "value"

    def test_get_option_missing(self):
        params = AgentParams()
        assert params.get_option("nonexistent") is None

    def test_get_option_with_default(self):
        params = AgentParams()
        assert params.get_option("nonexistent", "default") == "default"

    def test_has_columns_true(self):
        params = AgentParams(columns=["a"])
        assert params.has_columns() is True

    def test_has_columns_false(self):
        params = AgentParams()
        assert params.has_columns() is False

    def test_has_columns_empty_list(self):
        params = AgentParams(columns=[])
        assert params.has_columns() is False
