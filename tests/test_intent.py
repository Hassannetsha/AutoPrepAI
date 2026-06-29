"""Tests for Intent class."""

from business_logic.cleaning_coordinator.intent import Intent


class TestIntent:
    def test_default_initialization(self):
        intent = Intent(name="handle_missing_values")
        assert intent.name == "handle_missing_values"
        assert intent.columns == []
        assert intent.parameters == {}

    def test_with_columns_and_params(self):
        intent = Intent(
            name="remove_outliers",
            columns=["age", "salary"],
            parameters={"method": "iqr", "threshold": "1.5"}
        )
        assert intent.name == "remove_outliers"
        assert intent.columns == ["age", "salary"]
        assert intent.parameters == {"method": "iqr", "threshold": "1.5"}

    def test_has_column_true(self):
        intent = Intent(name="test", columns=["a", "b", "c"])
        assert intent.has_column("b") is True

    def test_has_column_false(self):
        intent = Intent(name="test", columns=["a", "b"])
        assert intent.has_column("z") is False

    def test_has_column_empty(self):
        intent = Intent(name="test")
        assert intent.has_column("anything") is False

    def test_get_parameter_existing(self):
        intent = Intent(name="test", parameters={"strategy": "mean"})
        assert intent.get_parameter("strategy") == "mean"

    def test_get_parameter_missing(self):
        intent = Intent(name="test")
        assert intent.get_parameter("strategy") is None

    def test_get_parameter_with_default(self):
        intent = Intent(name="test")
        assert intent.get_parameter("strategy", "mode") == "mode"

    def test_repr(self):
        intent = Intent(name="test", columns=["a"], parameters={"k": "v"})
        r = repr(intent)
        assert "Intent" in r
        assert "test" in r
        assert "a" in r
        assert "k" in r
