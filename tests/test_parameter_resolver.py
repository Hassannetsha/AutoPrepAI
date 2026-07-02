"""Tests for ParameterResolver."""

import pandas as pd
import pytest
from business_logic.cleaning_coordinator.data_context import DataContext
from business_logic.cleaning_coordinator.parameter_resolver import IntentColumnResolver
from business_logic.cleaning_coordinator.agent_params import AgentParams


class TestIntentColumnResolver:
    def test_resolve_with_tuple_intents(self):
        resolver = IntentColumnResolver(
            intent_names=["remove_outliers"],
            default_strategy="iqr"
        )
        ctx = DataContext(
            data=pd.DataFrame(),
            metadata={
                "intents": [
                    ["remove_outliers", "age", "salary"]
                ]
            }
        )
        params = resolver.resolve(ctx)
        assert isinstance(params, AgentParams)
        assert params.columns == ["age", "salary"]
        assert params.strategy == "iqr"
        assert params.options == {}

    def test_resolve_with_multiple_tuple_intents(self):
        resolver = IntentColumnResolver(
            intent_names=["handle_missing_values"],
            default_strategy="mean"
        )
        ctx = DataContext(
            data=pd.DataFrame(),
            metadata={
                "intents": [
                    ["handle_missing_values", "age", "salary"],
                    ["remove_outliers"]
                ]
            }
        )
        params = resolver.resolve(ctx)
        assert params.columns == ["age", "salary"]
        assert params.strategy == "mean"

    def test_resolve_with_nested_column_list(self):
        resolver = IntentColumnResolver(
            intent_names=["encode_categorical"],
            default_strategy="onehot"
        )
        ctx = DataContext(
            data=pd.DataFrame(),
            metadata={
                "intents": [
                    ["encode_categorical", ["color", "size"], "label"]
                ]
            }
        )
        params = resolver.resolve(ctx)
        assert params.columns == ["color", "size"]
        assert params.strategy == "onehot"

    def test_resolve_no_matching_intents(self):
        resolver = IntentColumnResolver(
            intent_names=["nonexistent"],
            default_strategy="default_strat"
        )
        ctx = DataContext(
            data=pd.DataFrame(),
            metadata={
                "intents": [("something_else",)]
            }
        )
        params = resolver.resolve(ctx)
        assert params.columns == []
        assert params.strategy == "default_strat"
        assert params.options == {}

    def test_resolve_empty_intents(self):
        resolver = IntentColumnResolver(
            intent_names=["test"],
            default_strategy="fallback"
        )
        ctx = DataContext(data=pd.DataFrame(), metadata={"intents": []})
        params = resolver.resolve(ctx)
        assert params.columns == []
        assert params.strategy == "fallback"

    def test_resolve_no_metadata_intents(self):
        resolver = IntentColumnResolver(
            intent_names=["test"],
            default_strategy="default"
        )
        ctx = DataContext(data=pd.DataFrame())
        params = resolver.resolve(ctx)
        assert params.columns == []
        assert params.strategy == "default"

    def test_resolve_strategy_uses_default(self):
        resolver = IntentColumnResolver(
            intent_names=["handle_missing_values"],
            default_strategy="mean"
        )
        ctx = DataContext(
            data=pd.DataFrame(),
            metadata={
                "intents": [
                    ["handle_missing_values"]
                ]
            }
        )
        params = resolver.resolve(ctx)
        assert params.strategy == "mean"

    def test_resolve_multiple_matching_intents_merge_columns(self):
        resolver = IntentColumnResolver(
            intent_names=["intent_a", "intent_b"],
            default_strategy="default"
        )
        ctx = DataContext(
            data=pd.DataFrame(),
            metadata={
                "intents": [
                    ("intent_a", "col1"),
                    ("intent_b", "col2")
                ]
            }
        )
        params = resolver.resolve(ctx)
        assert "col1" in params.columns
        assert "col2" in params.columns
