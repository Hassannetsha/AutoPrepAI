import io
import uuid
from datetime import date, datetime
import math
from pathlib import Path

import numpy as np
import pandas as pd

from data_context import DataContext
from pipeline_builder import PipelineBuilder


class MLPipelineService:
    OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
    TARGET_INTENTS = [
        "handle_missing_values",
        "detect_outliers",
        "remove_duplicates",
        "encode_categorical",
        "feature_selection",
        "fix_data_types",
        "remove_inconsistencies",
        "correct_spelling",
        "standardize_data",
        "scale_numerical",
        "feature_engineering",
    ]
    FULL_AUTO_INTENTS = [
        "handle_missing_values",
        "detect_outliers",
        "remove_duplicates",
        "remove_inconsistencies",
        "correct_spelling",
        "standardize_data",
        "feature_engineering",
    ]
    ALLOWED_MANUAL_INTENTS = {
        "fix_data_types",
        "remove_inconsistencies",
        "correct_spelling",
        "standardize_data",
        "remove_duplicates",
        "handle_missing_values",
        "remove_outliers",
        "keep_outliers",
        "select_features",
        "feature_selection",
        "scale_numerical",
        "encode_categorical",
        "suggest_features",
        "detect_outliers",
        "feature_engineering",
    }
    AUTO_COMMAND = (
        "clean data automatically: "
        "handle missing values with mean, remove duplicates, resolve inconsistencies, "
        "detect and remove outliers, correct spelling in categorical columns, "
        "engineer features for modeling"
    )

    @classmethod
    def _ensure_output_dir(cls) -> Path:
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return cls.OUTPUT_DIR

    @classmethod
    def save_processed_dataframe(cls, dataframe: pd.DataFrame) -> str:
        output_dir = cls._ensure_output_dir()
        filename = f"processed_{uuid.uuid4().hex}.csv"
        output_path = output_dir / filename
        dataframe.to_csv(output_path, index=False)
        return filename

    @classmethod
    def get_output_file_path(cls, filename: str) -> Path:
        if not filename:
            raise ValueError("Filename is required")

        output_dir = cls._ensure_output_dir().resolve()
        safe_name = Path(filename).name
        file_path = (output_dir / safe_name).resolve()

        if output_dir not in file_path.parents:
            raise ValueError("Invalid filename")
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError("File not found")

        return file_path
    #this function reads the uploaded file and converts it to a pandas dataframe
    # it supports both csv and excel files based on the file extension
    @staticmethod
    def dataframe_from_upload(file_bytes: bytes, filename: str | None = None) -> pd.DataFrame:
        if filename and filename.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(file_bytes))
        return pd.read_csv(io.BytesIO(file_bytes))

    @staticmethod
    def _normalize_mode(mode: str | None) -> str:
        raw_mode = (mode or "chat").strip().lower()
        mode_map = {
            "chat": "chat",
            "chat mode": "chat",
            "manual": "manual",
            "manual_selection": "manual",
            "manual selection mode": "manual",
            "full_auto": "full_auto",
            "full-auto": "full_auto",
            "auto": "full_auto",
            "full auto mode": "full_auto",
            "full": "full_auto",
        }
        normalized = mode_map.get(raw_mode)
        if normalized is None:
            raise ValueError("Invalid mode. Use one of: chat, manual, full_auto")
        return normalized

    @classmethod
    def _clean_manual_intents(cls, selected_intents: list[str] | None) -> list[str]:
        if not selected_intents:
            return []

        cleaned: list[str] = []
        for intent in selected_intents:
            intent_name = (intent or "").strip()
            if intent_name and intent_name in cls.ALLOWED_MANUAL_INTENTS and intent_name not in cleaned:
                cleaned.append(intent_name)
        return cleaned

    @staticmethod
    def _to_jsonable(value):
        if isinstance(value, dict):
            return {str(k): MLPipelineService._to_jsonable(v) for k, v in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [MLPipelineService._to_jsonable(item) for item in value]

        if isinstance(value, np.ndarray):
            return [MLPipelineService._to_jsonable(item) for item in value.tolist()]

        if isinstance(value, np.generic):
            return MLPipelineService._to_jsonable(value.item())

        if isinstance(value, (pd.Timestamp, datetime, date)):
            return value.isoformat()

        if isinstance(value, pd.Timedelta):
            return str(value)

        if value is None:
            return None

        if isinstance(value, bool):
            return value

        if isinstance(value, int):
            return value

        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value

        if isinstance(value, str):
            return value

        try:
            if pd.isna(value):
                return None
        except Exception:
            pass

        return str(value)

    @staticmethod
    def process_message(
        user_message: str,
        dataset_df: pd.DataFrame | None = None,
        mode: str = "chat",
        selected_intents: list[str] | None = None,
    ) -> dict:
        if dataset_df is None:
            raise ValueError("Dataset is required for processing")

        normalized_mode = MLPipelineService._normalize_mode(mode)

        context = DataContext(
            data=dataset_df.copy(),
            metadata={
                "has_text": True,
                "has_numeric": True,
                "has_categorical": True,
            },
        )

        effective_command = (user_message or "").strip() # for chat mode
        if normalized_mode == "manual":
            manual_intents = MLPipelineService._clean_manual_intents(selected_intents)
            if not manual_intents:
                manual_intents = MLPipelineService.TARGET_INTENTS.copy()
            context.metadata["nlp_done"] = True
            context.metadata["intents"] = [(intent,) for intent in manual_intents]
            effective_command = ""
        elif normalized_mode == "full_auto":
            context.metadata["nlp_done"] = True
            context.metadata["intents"] = [(intent,) for intent in MLPipelineService.FULL_AUTO_INTENTS]
            effective_command = MLPipelineService.AUTO_COMMAND
        elif not effective_command:
            raise ValueError("Message cannot be empty in chat mode")

        pipeline = PipelineBuilder.build_default_pipeline()
        final_context = pipeline.run(context=context, user_command=effective_command)
        output_file = MLPipelineService.save_processed_dataframe(final_context.data)

        result = {
            "shape": final_context.data.shape,
            "logs": final_context.logs,
            "metadata": final_context.metadata,
            "data_preview": final_context.data.head(10).to_dict(orient="records"),
            "output_file": output_file,
            "download_url": None,
        }
        return MLPipelineService._to_jsonable(result)
