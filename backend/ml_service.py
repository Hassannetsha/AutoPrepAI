import io
import uuid
from datetime import date, datetime
import math
from pathlib import Path

import numpy as np
import pandas as pd

from data_context import DataContext
from pipeline_builder import PipelineBuilder

from pipeline import Pipeline

from backend.b2_service import upload_file_to_b2

_cached_pipeline = None

def _get_pipeline() -> Pipeline:
    global _cached_pipeline
    if _cached_pipeline is None:
        _cached_pipeline = PipelineBuilder.build_default_pipeline()
    return _cached_pipeline

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
        conversation_id: str = "",
    ) -> dict:
        if dataset_df is None:
            raise ValueError("Dataset is required for processing")

        normalized_mode = MLPipelineService._normalize_mode(mode)
        effective_command = (user_message or "").strip()

        context = DataContext(
            data=dataset_df.copy(),
            metadata={
                "has_text": True,
                "has_numeric": True,
                "has_categorical": True,
                "user_command": effective_command,  # Store user_command early for NLP agent
            },
        )
        
        if normalized_mode == "full_auto":
            context.metadata["nlp_done"] = True
            context.metadata["intents"] = [(intent,) for intent in MLPipelineService.FULL_AUTO_INTENTS]
            effective_command = MLPipelineService.AUTO_COMMAND
        elif normalized_mode == "manual":
            cleaned = MLPipelineService._clean_manual_intents(selected_intents)
            if not cleaned:
                raise ValueError("No valid intents provided for manual mode")
            context.metadata["nlp_done"] = True
            context.metadata["intents"] = [(intent,) for intent in cleaned]
            if not effective_command:
                effective_command = f"Apply: {', '.join(cleaned)}"
        elif not effective_command:
            raise ValueError("Message cannot be empty in chat mode")
        
        # For chat mode: let NLP extract intents from the message
        # (nlp_done is NOT set, so NLP agent will run)

        print(f"[DEBUG] effective_command='{effective_command}'")
        print(f"[DEBUG] metadata user_command='{context.metadata.get('user_command')}'")

        pipeline = _get_pipeline()
        final_context = pipeline.run(context=context, user_command=effective_command)

        if normalized_mode == "chat":
            detected_intents = final_context.metadata.get("intents") or []
            if not detected_intents:
                raise ValueError(
                    "No preprocessing intent was detected from your chat message. "
                    "Please mention a data-cleaning action such as missing values, outliers, duplicates, scaling, or encoding."
                )

        output_file = MLPipelineService.save_processed_dataframe(final_context.data, conversation_id)

        result = {
            "shape": final_context.data.shape,
            "logs": final_context.logs,
            "metadata": final_context.metadata,
            "data_preview": final_context.data.head(10).to_dict(orient="records"),
            "output_file": output_file,
            "download_url": None,
        }
        jsonable = MLPipelineService._to_jsonable(result)
        jsonable["assistant_message"] = MLPipelineService._build_assistant_message(jsonable)
        return jsonable

    @classmethod
    def save_processed_dataframe(cls, dataframe: pd.DataFrame, conversation_id: str) -> str:
        filename = f"processed/{conversation_id}/{uuid.uuid4().hex}.csv"
        csv_bytes = dataframe.to_csv(index=False).encode("utf-8")
        upload_file_to_b2(csv_bytes, key=filename, content_type="text/csv")
        return filename

    @staticmethod
    def _build_assistant_message(result: dict) -> str:
        logs = result.get("logs") or []
        shape = result.get("shape")
        intents = result.get("metadata", {}).get("intents") or []
        # Extract intent names and consolidate related intents into friendly labels
        intent_names = []
        seen_groups = set()

        # Map specific intent identifiers to a canonical group
        intent_groups = {
            # Outliers
            "remove_outliers": "outliers",
            "detect_outliers": "outliers",
            # Missing values
            "handle_missing_values": "missing_values",
            # Duplicates
            "remove_duplicates": "duplicates",
            # Encoding
            "encode_categorical": "encoding",
            # Scaling
            "scale_numerical": "scaling",
            # Feature engineering / suggestion
            "suggest_features": "feature_engineering",
            "feature_engineering": "feature_engineering",
            # Feature selection
            "select_features": "feature_selection",
            "feature_selection": "feature_selection",
            # Data type fixes
            "fix_data_types": "data_types",
            "remove_inconsistencies": "data_types",
            # Spelling / standardization
            "correct_spelling": "spelling",
            "standardize_data": "standardization",
        }

        # Friendly display labels for groups
        display_labels = {
            "outliers": "Remove Outliers",
            "missing_values": "Handle Missing Values",
            "duplicates": "Remove Duplicates",
            "encoding": "Encoding",
            "scaling": "Scaling",
            "feature_engineering": "Feature Engineering",
            "feature_selection": "Feature Selection",
            "data_types": "Data Types",
            "spelling": "Spelling Correction",
            "standardization": "Standardization",
        }

        for intent in intents:
            name = intent[0] if isinstance(intent, (list, tuple)) and intent else str(intent)
            if not name:
                continue
            name = str(name).strip()
            if not name or name.lower() == "none":
                continue

            # canonicalize to lowercase key
            key = name.lower()
            group = intent_groups.get(key, key)

            # Only add one representative per group
            if group in seen_groups:
                continue
            seen_groups.add(group)

            # Use friendly label if available, otherwise prettify the raw intent
            label = display_labels.get(group)
            if not label:
                label = key.replace("_", " ").title()
            intent_names.append(label)

        parts = []

        if intent_names:
            parts.append(f"✅ Applied: **{', '.join(intent_names)}**.")

        if logs:
            parts.append("\n".join(f"• {log}" for log in logs))

        if shape:
            parts.append(f"📊 Dataset now has **{shape[0]} rows** and **{shape[1]} columns**.")

        return "\n\n".join(parts) if parts else "Processing completed successfully."
