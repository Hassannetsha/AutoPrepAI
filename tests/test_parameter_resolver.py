"""Tests for ParameterResolver."""

import pandas as pd
import pytest
from data_context import DataContext
from parameter_resolver import IntentColumnResolver
from agent_params import AgentParams
from intent import Intent


class TestIntentColumnResolver:
    def test_resolve_with_intent_objects(self):
        resolver = IntentColumnResolver(
            intent_names=["remove_outliers"],
            default_strategy="iqr"
        )
        ctx = DataContext(
            data=pd.DataFrame(),
            metadata={
                "intents": [
                    Intent(name="remove_outliers", columns=["age", "salary"],
                           parameters={"method": "zscore", "threshold": "3"})
                ]
            }
        )
        params = resolver.resolve(ctx)
        assert isinstance(params, AgentParams)
        assert params.columns == ["age", "salary"]
        assert params.strategy == "iqr"
        assert params.options == {"method": "zscore", "threshold": "3"}

    def test_resolve_with_tuple_intents(self):
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
                "intents": [Intent(name="something_else")]
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

    def test_resolve_strategy_from_parameters(self):
        resolver = IntentColumnResolver(
            intent_names=["handle_missing_values"],
            default_strategy="mean"
        )
        ctx = DataContext(
            data=pd.DataFrame(),
            metadata={
                "intents": [
                    Intent(name="handle_missing_values",
                           parameters={"strategy": "median"})
                ]
            }
        )
        params = resolver.resolve(ctx)
        assert params.strategy == "median"

    def test_resolve_multiple_matching_intents_merge_params(self):
        resolver = IntentColumnResolver(
            intent_names=["intent_a", "intent_b"],
            default_strategy="default"
        )
        ctx = DataContext(
            data=pd.DataFrame(),
            metadata={
                "intents": [
                    Intent(name="intent_a", columns=["col1"],
                           parameters={"a": "1"}),
                    Intent(name="intent_b", columns=["col2"],
                           parameters={"b": "2"})
                ]
            }
        )
        params = resolver.resolve(ctx)
        assert "col1" in params.columns
        assert "col2" in params.columns
        assert params.options.get("a") == "1"
        assert params.options.get("b") == "2"
