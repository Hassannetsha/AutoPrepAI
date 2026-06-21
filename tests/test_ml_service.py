import io
import math
import uuid
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backend.ml_service import MLPipelineService
from data_context import DataContext


class TestDataframeFromUpload:
    def test_csv_upload(self):
        csv_content = b"col1,col2\n1,a\n2,b\n"
        df = MLPipelineService.dataframe_from_upload(csv_content, "test.csv")
        assert list(df.columns) == ["col1", "col2"]
        assert len(df) == 2

    def test_csv_without_filename_raises_error(self):
        csv_content = b"a,b\n1,2\n"
        with pytest.raises(ValueError, match="File type is not supported"):
            MLPipelineService.dataframe_from_upload(csv_content)

    def test_excel_upload(self):
        df_orig = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        buf = io.BytesIO()
        df_orig.to_excel(buf, index=False)
        buf.seek(0)
        df = MLPipelineService.dataframe_from_upload(buf.read(), "data.xlsx")
        assert list(df.columns) == ["x", "y"]
        assert len(df) == 2


class TestNormalizeMode:
    @pytest.mark.parametrize(
        "input_mode,expected",
        [
            ("chat", "chat"),
            ("chat mode", "chat"),
            ("manual", "manual"),
            ("manual_selection", "manual"),
            ("full_auto", "full_auto"),
            ("full-auto", "full_auto"),
            ("auto", "full_auto"),
            ("full", "full_auto"),
            ("  Chat  ", "chat"),
        ],
    )
    def test_valid_modes(self, input_mode, expected):
        assert MLPipelineService._normalize_mode(input_mode) == expected

    @pytest.mark.parametrize("invalid", ["", "unknown", "something_else"])
    def test_invalid_modes(self, invalid):
        with pytest.raises(ValueError, match="Invalid mode"):
            MLPipelineService._normalize_mode(invalid)

    def test_none_mode(self):
        assert MLPipelineService._normalize_mode(None) == "chat"


class TestCleanManualIntents:
    @patch.object(MLPipelineService, "ALLOWED_MANUAL_INTENTS", new={"a", "b", "c"})
    def test_cleans_and_deduplicates(self):
        result = MLPipelineService._clean_manual_intents(["a", "b", "a", "  c  ", "invalid"])
        assert result == ["a", "b", "c"]

    def test_empty_input(self):
        assert MLPipelineService._clean_manual_intents([]) == []
        assert MLPipelineService._clean_manual_intents(None) == []


class TestToJsonable:
    def test_none(self):
        assert MLPipelineService._to_jsonable(None) is None

    def test_bool(self):
        assert MLPipelineService._to_jsonable(True) is True
        assert MLPipelineService._to_jsonable(False) is False

    def test_int(self):
        assert MLPipelineService._to_jsonable(42) == 42

    def test_float_nan_inf(self):
        assert MLPipelineService._to_jsonable(float("nan")) is None
        assert MLPipelineService._to_jsonable(float("inf")) is None
        assert MLPipelineService._to_jsonable(float("-inf")) is None
        assert MLPipelineService._to_jsonable(3.14) == 3.14

    def test_str(self):
        assert MLPipelineService._to_jsonable("hello") == "hello"

    def test_datetime_and_date(self):
        dt = datetime(2025, 1, 15, 12, 30, 0)
        assert MLPipelineService._to_jsonable(dt) == "2025-01-15T12:30:00"
        d = date(2025, 1, 15)
        assert MLPipelineService._to_jsonable(d) == "2025-01-15"

    def test_numpy_types(self):
        assert MLPipelineService._to_jsonable(np.int64(5)) == 5
        assert MLPipelineService._to_jsonable(np.float64(2.5)) == 2.5
        assert MLPipelineService._to_jsonable(np.array([1, 2, 3])) == [1, 2, 3]

    def test_dict(self):
        result = MLPipelineService._to_jsonable({"key": np.float64(1.0)})
        assert result == {"key": 1.0}

    def test_list(self):
        result = MLPipelineService._to_jsonable([1, "a", None])
        assert result == [1, "a", None]


class TestSaveAndGetOutputFile:
    def test_save_and_get_path(self, tmp_path):
        service = MLPipelineService()
        original_dir = MLPipelineService.OUTPUT_DIR
        try:
            MLPipelineService.OUTPUT_DIR = tmp_path
            df = pd.DataFrame({"a": [1, 2]})
            filename = MLPipelineService.save_processed_dataframe(df, "test_conv")
            assert isinstance(filename, str)
            assert filename.startswith("processed/test_conv/")
            assert filename.endswith(".csv")
            # Create file locally so get_output_file_path can find it
            file_path = tmp_path / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(file_path, index=False)
            path = MLPipelineService.get_output_file_path(filename)
            assert path.exists()
        finally:
            MLPipelineService.OUTPUT_DIR = original_dir

    def test_get_output_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            MLPipelineService.get_output_file_path("nonexistent.csv")

    def test_get_output_file_empty(self):
        with pytest.raises(ValueError, match="Filename is required"):
            MLPipelineService.get_output_file_path("")

    def test_get_output_file_path_traversal(self):
        with pytest.raises(ValueError, match="Invalid filename"):
            MLPipelineService.get_output_file_path("../../etc/passwd")


class TestBuildAssistantMessage:
    def test_with_intents(self):
        result = {
            "logs": ["Log 1", "Log 2"],
            "shape": [100, 5],
            "metadata": {"intents": [("handle_missing_values",)]},
        }
        msg = MLPipelineService._build_assistant_message(result)
        assert "Applied" in msg
        assert "Handle Missing Values" in msg
        assert "Log 1" in msg
        assert "100 rows" in msg

    def test_without_intents(self):
        result = {"logs": [], "shape": None, "metadata": {}}
        msg = MLPipelineService._build_assistant_message(result)
        assert msg == "Processing completed successfully."

    def test_empty_logs(self):
        result = {
            "logs": [],
            "shape": [50, 3],
            "metadata": {"intents": [("encode_categorical",)]},
        }
        msg = MLPipelineService._build_assistant_message(result)
        assert "Encoding" in msg
        assert "50 rows" in msg


class TestPrepareFullAuto:
    def test_sets_nlp_done_and_intents(self):
        df = pd.DataFrame({"a": [1]})
        context = DataContext(data=df, metadata={})
        ctx, cmd = MLPipelineService._prepare_full_auto(context, "user cmd")
        assert ctx.metadata["nlp_done"] is True
        assert len(ctx.metadata["intents"]) == len(MLPipelineService.FULL_AUTO_INTENTS)
        assert cmd == MLPipelineService.AUTO_COMMAND


class TestPrepareManual:
    @patch.object(MLPipelineService, "ALLOWED_MANUAL_INTENTS", new={"a", "b"})
    def test_sets_intents_from_selection(self):
        df = pd.DataFrame({"a": [1]})
        context = DataContext(data=df, metadata={})
        ctx, cmd = MLPipelineService._prepare_manual(context, "", ["a", "b"])
        assert ctx.metadata["nlp_done"] is True
        assert len(ctx.metadata["intents"]) == 2

    @patch.object(MLPipelineService, "ALLOWED_MANUAL_INTENTS", new={"a"})
    def test_raises_on_no_valid_intents(self):
        df = pd.DataFrame({"a": [1]})
        context = DataContext(data=df, metadata={})
        with pytest.raises(ValueError, match="No valid intents"):
            MLPipelineService._prepare_manual(context, "", ["invalid"])

    def test_provides_fallback_command(self):
        df = pd.DataFrame({"a": [1]})
        context = DataContext(data=df, metadata={})
        with patch.object(MLPipelineService, "ALLOWED_MANUAL_INTENTS", new={"a"}):
            ctx, cmd = MLPipelineService._prepare_manual(context, "", ["a"])
            assert "Apply:" in cmd


class TestPrepareChat:
    def test_requires_message(self):
        with pytest.raises(ValueError, match="Message cannot be empty"):
            MLPipelineService._prepare_chat("")

    def test_returns_command(self):
        assert MLPipelineService._prepare_chat("clean data") == "clean data"


class TestProcessMessage:
    @patch("backend.ml_service.utilities")
    @patch("backend.ml_service.PipelineBuilder")
    def test_raises_on_no_dataset(self, mock_builder, mock_utils):
        with pytest.raises(ValueError, match="Dataset is required"):
            MLPipelineService.process_message("hello", dataset_df=None)

    @patch("backend.ml_service.MLPipelineService.save_processed_dataframe")
    @patch("backend.ml_service.utilities")
    @patch("backend.ml_service.PipelineBuilder")
    def test_full_auto_execution(self, mock_builder, mock_utils, mock_save):
        mock_save.return_value = "output.csv"
        mock_utils.sessions = {}
        pipeline_mock = MagicMock()
        pipeline_mock.run.return_value = DataContext(
            data=pd.DataFrame({"a": [1]}),
            metadata={"intents": [("handle_missing_values",)]},
        )
        mock_builder.build_default_pipeline.return_value = pipeline_mock

        df = pd.DataFrame({"a": [1]})
        result, finished = MLPipelineService.process_message(
            "", dataset_df=df, mode="full_auto"
        )
        assert finished is True
        assert "assistant_message" in result

    @patch("backend.ml_service.MLPipelineService.save_processed_dataframe")
    @patch("backend.ml_service.utilities")
    @patch("backend.ml_service.PipelineBuilder")
    def test_manual_execution(self, mock_builder, mock_utils, mock_save):
        mock_save.return_value = "output.csv"
        mock_utils.sessions = {}
        pipeline_mock = MagicMock()
        pipeline_mock.run_single_agent.return_value = (
            DataContext(data=pd.DataFrame({"a": [1]}), metadata={"nlp_done": True}),
            False,
        )
        mock_builder.build_default_pipeline.return_value = pipeline_mock

        df = pd.DataFrame({"a": [1]})
        with patch.object(MLPipelineService, "ALLOWED_MANUAL_INTENTS", new={"test_intent"}):
            result, finished = MLPipelineService.process_message(
                "", dataset_df=df, mode="manual", selected_intents=["test_intent"]
            )
            assert finished is False

    @patch("backend.ml_service.MLPipelineService.save_processed_dataframe")
    @patch("backend.ml_service.utilities")
    @patch("backend.ml_service.PipelineBuilder")
    def test_chat_execution(self, mock_builder, mock_utils, mock_save):
        mock_save.return_value = "output.csv"
        mock_utils.sessions = {}
        pipeline_mock = MagicMock()
        pipeline_mock.run_single_agent.return_value = (
            DataContext(
                data=pd.DataFrame({"a": [1]}),
                metadata={"intents": [("handle_missing_values",)]},
            ),
            True,
        )
        mock_builder.build_default_pipeline.return_value = pipeline_mock

        df = pd.DataFrame({"a": [1]})
        result, finished = MLPipelineService.process_message(
            "handle missing values", dataset_df=df, mode="chat"
        )
        assert finished is True
