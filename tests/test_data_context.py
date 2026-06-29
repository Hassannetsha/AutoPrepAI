"""Tests for the DataContext class."""

import pandas as pd
import pytest

from business_logic.cleaning_coordinator.data_context import DataContext


class TestDataContext:
    def test_initialization(self):
        df = pd.DataFrame({"a": [1, 2]})
        ctx = DataContext(data=df)
        assert ctx.data is df
        assert ctx.metadata == {}
        assert ctx.logs == []
        assert ctx.user_id == ""
        assert ctx.conversation_id == ""

    def test_log(self):
        ctx = DataContext(data=pd.DataFrame())
        ctx.log("step 1")
        assert ctx.logs == ["step 1"]

    def test_get_metadata(self):
        ctx = DataContext(data=pd.DataFrame(), metadata={"key": "value"})
        assert ctx.get_metadata("key") == "value"
        assert ctx.get_metadata("nonexistent") is None

    def test_set_metadata(self):
        ctx = DataContext(data=pd.DataFrame())
        ctx.set_metadata("key", "value")
        assert ctx.metadata["key"] == "value"

    def test_get_and_set_data(self):
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"b": [2]})
        ctx = DataContext(data=df1)
        assert ctx.get_data() is df1
        ctx.set_data(df2)
        assert ctx.get_data() is df2
