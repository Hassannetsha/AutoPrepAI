"""Tests for EncodingAgent."""

import pandas as pd
from unittest.mock import MagicMock, patch

from business_logic.cleaning_coordinator.data_context import DataContext
from business_logic.cleaning_coordinator.agent_params import AgentParams


class TestEncodingAgent:
    @patch("agents.encoding_agent.EncoderFactory")
    @patch("agents.encoding_agent.detect_categorical_columns")
    def test_execute_auto_detect_columns(self, mock_detect, MockEncoderFactory):
        from ml_layer.agents.encoding_agent import EncodingAgent

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = pd.DataFrame({
            "color_red": [1, 0], "color_blue": [0, 1]
        })
        MockEncoderFactory.get_encoder.return_value = mock_encoder
        mock_detect.return_value = ["color"]

        agent = EncodingAgent()
        df = pd.DataFrame({"color": ["red", "blue"], "val": [1, 2]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=[], strategy="onehot")

        result = agent.execute(ctx, params)
        assert result.metadata.get("encoded") is True

    @patch("agents.encoding_agent.EncoderFactory")
    def test_execute_specific_columns(self, MockEncoderFactory):
        from ml_layer.agents.encoding_agent import EncodingAgent

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = pd.DataFrame({
            "size_small": [1, 0], "size_large": [0, 1], "val": [1, 2]
        })
        MockEncoderFactory.get_encoder.return_value = mock_encoder

        agent = EncodingAgent()
        df = pd.DataFrame({"size": ["small", "large"], "val": [1, 2]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=["size"], strategy="onehot")

        result = agent.execute(ctx, params)
        assert result.metadata.get("encoded") is True

    @patch("agents.encoding_agent.EncoderFactory")
    def test_execute_method_from_columns(self, MockEncoderFactory):
        from ml_layer.agents.encoding_agent import EncodingAgent

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = pd.DataFrame({"color": [0, 1]})
        MockEncoderFactory.get_encoder.return_value = mock_encoder

        agent = EncodingAgent()
        df = pd.DataFrame({"color": ["red", "blue"]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=[["color"], "label"], strategy="onehot")

        result = agent.execute(ctx, params)
        MockEncoderFactory.get_encoder.assert_called_with("label")

    def test_execute_no_categorical_columns(self):
        from ml_layer.agents.encoding_agent import EncodingAgent

        agent = EncodingAgent()
        df = pd.DataFrame({"x": [1, 2, 3]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=[], strategy="onehot")

        result = agent.execute(ctx, params)
        assert "encoded" not in result.metadata

    @patch("agents.encoding_agent.EncoderFactory")
    def test_execute_with_target_column(self, MockEncoderFactory):
        from ml_layer.agents.encoding_agent import EncodingAgent

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = pd.DataFrame({
            "color_red": [1, 0], "color_blue": [0, 1], "price": [10, 20]
        })
        MockEncoderFactory.get_encoder.return_value = mock_encoder

        agent = EncodingAgent()
        df = pd.DataFrame({"color": ["red", "blue"], "price": [10, 20]})
        ctx = DataContext(data=df)
        params = AgentParams(
            columns=[{"target": "price"}, "color"],
            strategy="target"
        )

        result = agent.execute(ctx, params)
        assert result.metadata.get("encoded") is True

    @patch("agents.encoding_agent.EncoderFactory")
    def test_execute_error_logged(self, MockEncoderFactory):
        from ml_layer.agents.encoding_agent import EncodingAgent

        mock_encoder = MagicMock()
        mock_encoder.encode.side_effect = Exception("encode failed")
        MockEncoderFactory.get_encoder.return_value = mock_encoder

        agent = EncodingAgent()
        df = pd.DataFrame({"color": ["red", "blue"]})
        ctx = DataContext(data=df)
        params = AgentParams(columns=["color"], strategy="onehot")

        result = agent.execute(ctx, params)
        assert "encoded" not in result.metadata
