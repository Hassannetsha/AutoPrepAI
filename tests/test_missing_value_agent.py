"""Tests for MissingValueAgent."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from business_logic.cleaning_coordinator.data_context import DataContext
from business_logic.cleaning_coordinator.agent_params import AgentParams


class TestMissingValueAgent:
    def test_execute_no_columns_and_no_strategy_auto_selects(self):
        from ml_layer.agents.missing_value_agent import MissingValueAgent

        agent = MissingValueAgent()
        df = pd.DataFrame({"numeric_col": [1.0, 2.0, np.nan, 4.0]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=[], strategy="")

        result = agent.execute(ctx, params)

        assert isinstance(result.data, pd.DataFrame)
        assert result.data.isnull().sum().sum() == 0

    def test_execute_with_mean_strategy(self):
        from ml_layer.agents.missing_value_agent import MissingValueAgent

        agent = MissingValueAgent()
        df = pd.DataFrame({"age": [25.0, np.nan, 30.0, 35.0, np.nan]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=["age"], strategy="mean")

        result = agent.execute(ctx, params)

        assert result.data.isnull().sum().sum() == 0
        assert result.data["age"].iloc[1] == pytest.approx(30.0, abs=1)

    def test_execute_with_median_strategy(self):
        from ml_layer.agents.missing_value_agent import MissingValueAgent

        agent = MissingValueAgent()
        df = pd.DataFrame({"age": [25.0, np.nan, 30.0, 35.0, np.nan]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=["age"], strategy="median")

        result = agent.execute(ctx, params)

        assert result.data.isnull().sum().sum() == 0
        assert result.data["age"].iloc[1] == 30.0

    def test_execute_categorical_mode_fill(self):
        from ml_layer.agents.missing_value_agent import MissingValueAgent

        agent = MissingValueAgent()
        df = pd.DataFrame({"category": ["a", "a", np.nan, "b", np.nan]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=["category"], strategy="mode")

        result = agent.execute(ctx, params)

        assert result.data.isnull().sum().sum() == 0
        assert result.data["category"].iloc[2] == "a"

    def test_execute_no_valid_columns(self):
        from ml_layer.agents.missing_value_agent import MissingValueAgent

        agent = MissingValueAgent()
        df = pd.DataFrame()
        ctx = DataContext(data=df)
        params = AgentParams(columns=[])

        result = agent.execute(ctx, params)
        assert len(result.data.columns) == 0

    def test_execute_strategy_from_intent_override(self):
        from ml_layer.agents.missing_value_agent import MissingValueAgent

        agent = MissingValueAgent()
        df = pd.DataFrame({"age": [25.0, np.nan, 30.0]})
        ctx = DataContext(
            data=df,
            metadata={
                "intents": [["handle_missing_values", "age", "median"]]
            }
        )
        params = AgentParams(columns=["age"], strategy="mean")

        result = agent.execute(ctx, params)
        assert result.data.isnull().sum().sum() == 0

    def test_execute_knn_strategy(self):
        from ml_layer.agents.missing_value_agent import MissingValueAgent

        agent = MissingValueAgent()
        np.random.seed(42)
        df = pd.DataFrame({
            "a": [1.0, 2.0, np.nan, 4.0, 5.0],
            "b": [2.0, np.nan, 4.0, 5.0, 6.0],
        })
        ctx = DataContext(data=df)
        params = AgentParams(columns=["a", "b"], strategy="knn")

        result = agent.execute(ctx, params)
        assert result.data.isnull().sum().sum() == 0

    def test_execute_fallback_to_missing_values_demo(self):
        from ml_layer.agents.missing_value_agent import MissingValueAgent

        agent = MissingValueAgent()
        df = pd.DataFrame({"age": [25.0, np.nan, 30.0]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=["age"], strategy="unknown_strat")

        result = agent.execute(ctx, params)
        assert result.data.isnull().sum().sum() == 0
