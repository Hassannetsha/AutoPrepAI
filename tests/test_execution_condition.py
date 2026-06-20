"""Tests for execution conditions."""

import pytest
import pandas as pd
from data_context import DataContext
from execution_condition import AlwaysTrueCondition, AlwaysFalseCondition, IntentBasedCondition
from intent import Intent


class TestAlwaysTrueCondition:
    def test_always_true(self):
        condition = AlwaysTrueCondition()
        ctx = DataContext(data=pd.DataFrame())
        assert condition.evaluate(ctx) is True


class TestAlwaysFalseCondition:
    def test_always_false(self):
        condition = AlwaysFalseCondition()
        ctx = DataContext(data=pd.DataFrame())
        assert condition.evaluate(ctx) is False


class TestIntentBasedCondition:
    def test_any_operator_with_intent_objects(self):
        condition = IntentBasedCondition(["remove_duplicates", "scale_numerical"], operator="any")
        ctx = DataContext(
            data=pd.DataFrame(),
            metadata={"intents": [Intent(name="remove_duplicates")]}
        )
        assert condition.evaluate(ctx) is True

    def test_any_operator_with_tuple_intents(self):
        condition = IntentBasedCondition(["handle_missing_values"], operator="any")
        ctx = DataContext(
            data=pd.DataFrame(),
            metadata={"intents": [["handle_missing_values", "col1", "mean"]]}
        )
        assert condition.evaluate(ctx) is True

    def test_any_operator_with_string_intents(self):
        condition = IntentBasedCondition(["encode_categorical"], operator="any")
        ctx = DataContext(
            data=pd.DataFrame(),
            metadata={"intents": ["encode_categorical"]}
        )
        assert condition.evaluate(ctx) is True

    def test_any_operator_no_match(self):
        condition = IntentBasedCondition(["remove_outliers"], operator="any")
        ctx = DataContext(
            data=pd.DataFrame(),
            metadata={"intents": [Intent(name="handle_missing_values")]}
        )
        assert condition.evaluate(ctx) is False

    def test_all_operator_all_match(self):
        condition = IntentBasedCondition(["remove_duplicates", "scale_numerical"], operator="all")
        ctx = DataContext(
            data=pd.DataFrame(),
            metadata={"intents": [
                Intent(name="remove_duplicates"),
                Intent(name="scale_numerical")
            ]}
        )
        assert condition.evaluate(ctx) is True

    def test_all_operator_partial_match(self):
        condition = IntentBasedCondition(["remove_duplicates", "scale_numerical"], operator="all")
        ctx = DataContext(
            data=pd.DataFrame(),
            metadata={"intents": [Intent(name="remove_duplicates")]}
        )
        assert condition.evaluate(ctx) is False

    def test_empty_intents(self):
        condition = IntentBasedCondition(["anything"], operator="any")
        ctx = DataContext(data=pd.DataFrame(), metadata={"intents": []})
        assert condition.evaluate(ctx) is False

    def test_no_intents_in_metadata(self):
        condition = IntentBasedCondition(["anything"], operator="any")
        ctx = DataContext(data=pd.DataFrame())
        assert condition.evaluate(ctx) is False

    def test_invalid_operator_raises(self):
        with pytest.raises(ValueError, match="Operator must be 'any' or 'all'"):
            IntentBasedCondition(["test"], operator="invalid")

    def test_mixed_intent_formats(self):
        condition = IntentBasedCondition(["encode_categorical", "handle_missing_values"], operator="any")
        ctx = DataContext(
            data=pd.DataFrame(),
            metadata={"intents": [
                Intent(name="encode_categorical", columns=["color"]),
                ["handle_missing_values", "age"],
                "scale_numerical"
            ]}
        )
        assert condition.evaluate(ctx) is True

    def test_all_operator_empty_intents(self):
        condition = IntentBasedCondition(["a", "b"], operator="all")
        ctx = DataContext(data=pd.DataFrame(), metadata={"intents": []})
        assert condition.evaluate(ctx) is False
