import os
import re
import time
from typing import List, Optional, Dict, Any

import streamlit as st
import pandas as pd
import dspy
from ml_layer.features.engineering_service import *
import json
from business_logic.services.api_key_manager import get_key_manager, get_api_key, rotate_api_key

# Initialize the key manager (singleton)
_key_manager = get_key_manager()

class NLPService:
    """AutoPrepAI application class: encapsulates DSPy setup, pipeline creation, data loading and Streamlit UI.

    Design notes to respect SOLID:
    - Single Responsibility: each method has one clear responsibility (setup, load data, build pipeline, render UI).
    - Open/Closed: pipeline construction can accept training data and demos without modifying the class.
    - Liskov Substitution: public API is stable (run()).
    - Interface Segregation: consumer (new.py) uses only run(); internals are split into helpers.
    - Dependency Inversion: external deps (env, streamlit, dspy) are injected via environment or parameters where practical.
    """
    
    _lm = None
    _pipeline = None
    def __init__(self, groq_api_key=None):
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        
        # Initialize shared resources once
        if NLPService._lm is None:
            self._init_lm()
        self.lm = NLPService._lm
        if NLPService._pipeline is None:
            self._init_pipeline()

    @staticmethod
    def _validate_key(key: str) -> bool:
        import litellm
        try:
            litellm.completion(
                model="groq/llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "ping"}],
                api_key=key,
                max_tokens=5,
                temperature=0,
            )
            return True
        except Exception:
            return False

    def _init_lm(self):
        max_keys = _key_manager.get_total_keys_count()
        for _ in range(max_keys):
            api_key = _key_manager.get_current_key()
            if self._validate_key(api_key):
                break
            print(f"⚠️ Key #{_key_manager.current_index + 1} invalid, trying next...")
            _key_manager.mark_key_failed()
            try:
                _key_manager.rotate_key()
            except RuntimeError:
                raise RuntimeError("All API keys are invalid. Please update ApiKeys.txt with valid Groq keys.")
        else:
            raise RuntimeError("All API keys are invalid. Please update ApiKeys.txt with valid Groq keys.")

        os.environ["GROQ_API_KEY"] = api_key
        lm = dspy.LM(
            model="groq/llama-3.3-70b-versatile",
            api_key=api_key,
            max_tokens=1000
        )
        dspy.settings.configure(lm=lm)
        NLPService._lm = lm
        self.lm = lm
        print(f"✅ Using API Key #{_key_manager.current_index + 1}/{_key_manager.get_total_keys_count()}")

    def _init_pipeline(self):
        NLPService._pipeline = self.build_pipeline()

    def build_pipeline(self):
        return NLPService.OptimizedIntentPipeline()

    # ---------- DSPy Signatures ----------
    # Keep these as inner classes for encapsulation but accessible by dspy
    class SplitIntoTasks(dspy.Signature):
        """Split a complex user command into distinct preprocessing steps.
        Rules:
        - If the user mentions multiple operations (e.g., 'handle missing values and remove outliers'),
            split into one line per operation.
        - If the user mentions columns (e.g., 'for age and salary'), keep them in the same line.
        - If no columns are mentioned, keep it as a single task that applies to all columns.
        """
        user_command = dspy.InputField(desc="The user command (one or more preprocessing actions)")
        dataset_columns = dspy.InputField(desc="Available dataset columns (comma-separated)", default="")
        tasks = dspy.OutputField(desc="Each preprocessing action on a separate line.")

    class ClassifyIntent(dspy.Signature):
        """Classify the data preprocessing intent from a task description.

        CRITICAL DISAMBIGUATION:
        - "standardize the data" / "standardize" / "make data standardized" = standardize_data (unify CATEGORICAL/TEXT values like names, department codes, date formats, city names). This is about TEXT/CATEGORICAL consistency. NEVER scale_numerical.
        - "scale the data" / "scale numerical" / "normalize numeric" = scale_numerical (transform NUMBERS to a range via StandardScaler, MinMax, z-score). This is about NUMERICAL transformation. NEVER standardize_data.

        Available intents:
        - handle_missing_values: Fill, impute, or handle missing/null/NaN values
        - detect_outliers / remove_outliers: Identify and remove outliers, anomalies, or extreme values
        - keep_outliers: Preserve or keep outliers in the data
        - remove_duplicates: Remove duplicate rows or entries
        - encode_categorical: Convert categorical/text columns to numeric via one-hot, label, or frequency encoding
        - feature_selection / select_features: Select important features or columns for modeling
        - fix_data_types / remove_inconsistencies: Detect and resolve inconsistent types (dates, numbers, booleans)
        - correct_spelling: Fix spelling errors in categorical/text columns
        - standardize_data: Standardize CATEGORICAL/TEXT values to consistent formats — e.g., 'HR'→'Human Resources', 'NY'→'New York', 'Jan'→'January'. For unifying inconsistent text entries across rows. This is NOT for numerical scaling.
        - scale_numerical: Scale NUMERICAL columns to a common range (StandardScaler, MinMax, RobustScaler, z-score). For transforming numbers. This is NOT for standardizing text/categorical data.
        - feature_engineering / suggest_features: Suggest and/or apply new derived features from existing data
        - unknown_intent: Any request that is NOT about data preprocessing — e.g. cooking, cleaning the house, playing games, weather, jokes, general chat, or anything unrelated to cleaning or transforming a dataset.
          WARNING: Just because a request contains the word "clean", "top", "fix", "prepare", or "handle" does NOT make it data preprocessing.
          Words like "clean the top", "clean my room", "clean the table", "wash the dishes", "fix the car", "cook dinner", "play music", "weather today" are NOT data preprocessing — classify them as unknown_intent.
        """
        task = dspy.InputField(desc="A single preprocessing task description")
        intent = dspy.OutputField(desc="The intent category (must be one of: handle_missing_values, detect_outliers, remove_outliers, keep_outliers, remove_duplicates, encode_categorical, feature_selection, select_features, fix_data_types, remove_inconsistencies, correct_spelling, standardize_data, scale_numerical, feature_engineering, suggest_features, unknown_intent). REMEMBER: 'standardize data' = standardize_data (CATEGORICAL formatting), 'scale data' = scale_numerical (NUMERICAL transformation).")
        confidence = dspy.OutputField(desc="Confidence score between 0.0 and 1.0")
        reasoning = dspy.OutputField(desc="Brief explanation for the classification")

    class ExtractParameters(dspy.Signature):
        """Extract specific parameters from a preprocessing task.

        Look for:
        - Column names mentioned in the task (e.g., "for column X", "X and Y", "features related to X").
        - If no specific columns are mentioned, set columns='none' (it will mean all columns later).
        - Methods like: mean, median, mode, IQR, z-score, one-hot, label encoding
        - Numeric values or thresholds
        - For feature_selection / select_features, list only column names that appear in the task.
          Do NOT add column names from the dataset_columns that aren't mentioned in the task.
        """
        task = dspy.InputField(desc="A preprocessing task description")
        dataset_columns = dspy.InputField(desc="Available column names (comma-separated)", default="")
        intent = dspy.InputField(desc="The classified intent")
        columns = dspy.OutputField(desc="Column names mentioned (comma-separated), or 'none' if not specified. For feature_selection / select_features intent: extract ONLY column names that appear in the task text itself. NEVER invent column names from dataset_columns that are not explicitly mentioned in the task. If the user says 'target column: X' or 'for target column: X', extract X. If the user says 'features related to X and Y' or 'select features to columns: X, Y', extract only X and Y (not the word 'target'). The word 'target' from dataset_columns should never be added unless the task literally says the word 'target'.")
        method = dspy.OutputField(desc="Method/algorithm mentioned (e.g., mean, median, mode), or 'none'")
        other_params = dspy.OutputField(desc="Other parameters as key:value pairs (comma-separated), or 'none'")
    
    class OptimizedIntentPipeline(dspy.Module):
        """Intent understanding pipeline with few-shot learning from training data"""

        # Keyword-based classification — checked BEFORE DSPy for speed & reliability.
        # Each key has substrings; if ANY match the task (lowered), the intent is returned directly.
        INTENT_KEYWORDS = {
            "handle_missing_values": ["missing", "null value", "nan", "imput", "fill na", "fill null", "empty value"],
            "detect_outliers":       ["detect outlier", "find outlier", "identify outlier", "detect anomaly", "find anomaly", "anomali"],
            "remove_outliers":       ["remove outlier", "delete outlier", "eliminate outlier", "drop outlier", "remove extreme", "delete extreme"],
            "keep_outliers":         ["keep outlier", "preserve outlier", "retain outlier", "don't remove outlier", "don't delete outlier", "save outlier"],
            "remove_duplicates":     ["remove duplicate", "delete duplicate", "drop duplicate", "deduplicat", "remove repeated", "duplicate row"],
            "encode_categorical":    ["encode categorical", "encode category", "encode column", "dummy variable", "one-hot", "one hot", "label encode", "categorical encode"],
            "feature_selection":     ["feature select", "select feature", "select important", "choose feature", "select relevant", "most important feature", "top feature"],
            "fix_data_types":        ["fix data type", "fix type", "data type convers", "type mismatch", "correct type", "type correct", "remove inconsist", "inconsist"],
            "correct_spelling":      ["spelling", "misspell", "typo", "correct spell", "spell check", "fix spell"],
            "scale_numerical":       ["scale the ", "scale all ", "scale every ", "scale numeric",
                                      "scale column", "scale age", "scale income", "scale salary",
                                      "scale price", "scale rating", "scale year", "scaler agent",
                                      "use scaler", "use the scaler", "run scaler", "apply scaler",
                                      "normalize numeric", "normalize numerical", "normalize continuous",
                                      "z-score", "minmax", "min-max", "standard scaler",
                                      "rescale", "transform numeric", "transform numerical"],
            "standardize_data":      ["standardiz", "standariz", "stnadardiz", "stabdariz", "stdize",
                                      "homogeniz", "harmoniz", "categorical standard", "clean up value",
                                      "clean up text", "string standard"],
            "feature_engineering":   ["feature engineer", "engineer feature", "create new feature",
                                      "generate feature", "suggest feature", "suggest new feature",
                                      "derive feature", "new column from", "new features"],
        }

        def __init__(self):
            super().__init__()
            self.split_tasks = dspy.ChainOfThought(NLPService.SplitIntoTasks)
            self.classify = dspy.ChainOfThought(NLPService.ClassifyIntent)
            self.extract_params = dspy.ChainOfThought(NLPService.ExtractParameters)
            self.suggest_features = dspy.ChainOfThought(SuggestFeatures)
            self._setup_few_shot_examples()

        METHOD_KEYWORDS = {
            "mean": ["mean", "average"],
            "median": ["median"],
            "mode": ["mode", "most frequent"],
            "constant": ["constant", "fill with", "replace with"],
            "knn": ["knn", "k-nearest", "nearest neighbor"],
            "iterative": ["iterative", "mice"],
            "minmax": ["minmax", "min-max"],
            "one-hot": ["one-hot", "one hot", "dummy"],
            "label": ["label encoding", "label encode", "label"],
        }

        @staticmethod
        def _extract_method_from_text(task: str):
            """Extract method/strategy from task text (e.g. 'using median' → 'median')."""
            task_lower = task.lower().strip()
            for method, keywords in NLPService.OptimizedIntentPipeline.METHOD_KEYWORDS.items():
                for kw in keywords:
                    if kw in task_lower:
                        return method
            return None

        def _keyword_classify(self, task: str):
            """Check task against INTENT_KEYWORDS; return intent if matched, else None."""
            task_lower = task.lower().strip()
            for intent, keywords in self.INTENT_KEYWORDS.items():
                for kw in keywords:
                    if kw in task_lower:
                        return intent
            return None

        def _extract_columns_from_task(self, task: str, dataset_columns: str) -> list:
            """Extract column names from task text that exist in dataset_columns.

            Uses word-boundary matching so 'age' doesn't match 'page'.
            Handles common English plurals (e.g. 'departments' ↔ 'Department', 'categories' ↔ 'Category').
            Returns a list of matched column names (preserving original case from dataset_columns).
            """
            if not dataset_columns:
                return []
            task_lower = task.lower().strip()
            found = []
            for col in [c.strip() for c in dataset_columns.split(",") if c.strip()]:
                col_lower = col.lower()
                variants = {col_lower}
                # +s plural (departments → Department)
                variants.add(col_lower + "s")
                # -ies → -y plural (categories → Category)
                if col_lower.endswith("y"):
                    variants.add(col_lower[:-1] + "ies")
                # Strip trailing s (columns → column)
                if col_lower.endswith("s"):
                    variants.add(col_lower[:-1])
                # -y → -ies (Category → categories)
                if col_lower.endswith("ies"):
                    variants.add(col_lower[:-3] + "y")
                if any(re.search(rf'(?<!\w){re.escape(v)}(?!\w)', task_lower) for v in variants):
                    found.append(col)
            return found

        def _setup_few_shot_examples(self):
            """Setup curated few-shot examples — no random sampling, always representative."""
            # ── ClassifyIntent demos: curated 2 per intent ───────────────
            # unknown_intent (hard-coded, shown first so DSPy sees them first)
            unknown_demos = [
                dspy.Example(task="clean the head",           intent="unknown_intent", confidence="0.98", reasoning="washing one's head/hair, not data cleaning").with_inputs('task'),
                dspy.Example(task="clean the top for me please", intent="unknown_intent", confidence="0.98", reasoning="cleaning a physical table top, not data cleaning").with_inputs('task'),
                dspy.Example(task="clean the floor",          intent="unknown_intent", confidence="0.98", reasoning="sweeping a physical floor, not data cleaning").with_inputs('task'),
                dspy.Example(task="clean my room",            intent="unknown_intent", confidence="0.98", reasoning="tidying a physical room, not a dataset").with_inputs('task'),
                dspy.Example(task="organize my room",         intent="unknown_intent", confidence="0.98", reasoning="organizing a physical room, not data").with_inputs('task'),
                dspy.Example(task="prepare lunch",            intent="unknown_intent", confidence="0.98", reasoning="cooking food, not data preparation").with_inputs('task'),
                dspy.Example(task="fix the car",              intent="unknown_intent", confidence="0.98", reasoning="fixing a vehicle, not data fixing").with_inputs('task'),
                dspy.Example(task="wash the dishes",          intent="unknown_intent", confidence="0.98", reasoning="washing dishes, not data cleaning").with_inputs('task'),
                dspy.Example(task="clean the car",            intent="unknown_intent", confidence="0.98", reasoning="washing a vehicle, not data cleaning").with_inputs('task'),
                dspy.Example(task="clean the kitchen",        intent="unknown_intent", confidence="0.98", reasoning="cleaning a physical kitchen, not data preprocessing").with_inputs('task'),
            ]
            # Disambiguation demos (shown before agent demos so DSPy learns the distinction first)
            disambiguate_demos = [
                dspy.Example(task="standardize the data",     intent="standardize_data", confidence="0.98", reasoning="Standardizing data means unifying categorical/text formats, not numerical scaling").with_inputs('task'),
                dspy.Example(task="scale the data",           intent="scale_numerical",  confidence="0.98", reasoning="Scaling data means transforming numerical values to a range").with_inputs('task'),
            ]
            # Curated agent demos — 2 per intent, hand-picked for clarity
            agent_demos = [
                dspy.Example(task="handle missing values",    intent="handle_missing_values", confidence="0.95", reasoning="asks to handle/fill null values").with_inputs('task'),
                dspy.Example(task="fill null values in age column", intent="handle_missing_values", confidence="0.95", reasoning="asks to fill null/NaN in a column").with_inputs('task'),
                dspy.Example(task="detect outliers in the data", intent="detect_outliers", confidence="0.95", reasoning="asks to find outlier/extreme values").with_inputs('task'),
                dspy.Example(task="find extreme values",     intent="detect_outliers", confidence="0.95", reasoning="asks to detect anomalous data points").with_inputs('task'),
                dspy.Example(task="remove outliers",          intent="remove_outliers", confidence="0.95", reasoning="asks to delete outlier rows").with_inputs('task'),
                dspy.Example(task="delete extreme values from dataset", intent="remove_outliers", confidence="0.95", reasoning="asks to remove extreme values").with_inputs('task'),
                dspy.Example(task="keep outliers",            intent="keep_outliers", confidence="0.95", reasoning="asks to preserve outlier rows").with_inputs('task'),
                dspy.Example(task="don't remove outliers",    intent="keep_outliers", confidence="0.95", reasoning="asks to NOT delete outliers").with_inputs('task'),
                dspy.Example(task="remove duplicate rows",    intent="remove_duplicates", confidence="0.95", reasoning="asks to deduplicate the dataset").with_inputs('task'),
                dspy.Example(task="deduplicate the dataset",  intent="remove_duplicates", confidence="0.95", reasoning="asks to remove repeated entries").with_inputs('task'),
                dspy.Example(task="encode categorical columns", intent="encode_categorical", confidence="0.95", reasoning="asks to convert categories to numbers").with_inputs('task'),
                dspy.Example(task="one-hot encode category column", intent="encode_categorical", confidence="0.95", reasoning="asks to one-hot encode categorical data").with_inputs('task'),
                dspy.Example(task="select important features", intent="feature_selection", confidence="0.95", reasoning="asks to choose relevant columns for modeling").with_inputs('task'),
                dspy.Example(task="choose relevant columns for modeling", intent="feature_selection", confidence="0.95", reasoning="asks to select features/columns").with_inputs('task'),
                dspy.Example(task="fix data types",           intent="fix_data_types", confidence="0.95", reasoning="asks to correct column data types").with_inputs('task'),
                dspy.Example(task="remove inconsistencies in data", intent="fix_data_types", confidence="0.95", reasoning="asks to resolve inconsistent values").with_inputs('task'),
                dspy.Example(task="correct spelling mistakes", intent="correct_spelling", confidence="0.95", reasoning="asks to fix spelling errors").with_inputs('task'),
                dspy.Example(task="fix typos in text columns", intent="correct_spelling", confidence="0.95", reasoning="asks to correct misspellings").with_inputs('task'),
                dspy.Example(task="standardize categorical values", intent="standardize_data", confidence="0.95", reasoning="asks to unify categorical text formats").with_inputs('task'),
                dspy.Example(task="standardize string columns", intent="standardize_data", confidence="0.95", reasoning="asks to standardize text/string data").with_inputs('task'),
                dspy.Example(task="scale numerical columns",  intent="scale_numerical", confidence="0.95", reasoning="asks to scale/transform numeric columns").with_inputs('task'),
                dspy.Example(task="normalize numeric data with standard scaler", intent="scale_numerical", confidence="0.95", reasoning="asks to normalize numerical variables").with_inputs('task'),
                dspy.Example(task="create new features from existing data", intent="feature_engineering", confidence="0.95", reasoning="asks to engineer new features").with_inputs('task'),
                dspy.Example(task="suggest new features for modeling", intent="feature_engineering", confidence="0.95", reasoning="asks to suggest derived features").with_inputs('task'),
            ]
            self.classify.demos = unknown_demos + disambiguate_demos + agent_demos

            # ── ExtractParameters demos ─────────────────────────────────────
            self.extract_params.demos = [
                # ── feature_selection / select_features ─────────────────
                dspy.Example(
                    task="select features for target=Churn",
                    dataset_columns="age, income, education_years, Churn",
                    intent="feature_selection",
                    columns="Churn",
                    method="none",
                    other_params="none",
                ).with_inputs("task", "dataset_columns", "intent"),
                dspy.Example(
                    task="select top 10 features for predicting income",
                    dataset_columns="age, income, education_years, Churn",
                    intent="feature_selection",
                    columns="income",
                    method="none",
                    other_params="top:10",
                ).with_inputs("task", "dataset_columns", "intent"),
                dspy.Example(
                    task="feature selection for target column: text_review",
                    dataset_columns="age, income, text_review, id",
                    intent="feature_selection",
                    columns="text_review",
                    method="none",
                    other_params="none",
                ).with_inputs("task", "dataset_columns", "intent"),
                dspy.Example(
                    task="select relevant features related to income",
                    dataset_columns="age, income, education_years, Churn",
                    intent="feature_selection",
                    columns="income",
                    method="none",
                    other_params="none",
                ).with_inputs("task", "dataset_columns", "intent"),
                dspy.Example(
                    task="pick features that predict churn",
                    dataset_columns="age, income, education_years, churn",
                    intent="feature_selection",
                    columns="churn",
                    method="none",
                    other_params="none",
                ).with_inputs("task", "dataset_columns", "intent"),
                dspy.Example(
                    task="select features target column: rating",
                    dataset_columns="age, income, rating, id",
                    intent="feature_selection",
                    columns="rating",
                    method="none",
                    other_params="none",
                ).with_inputs("task", "dataset_columns", "intent"),
                dspy.Example(
                    task="select related features to columns: city and text_review",
                    dataset_columns="id, age, income, education_years, target, city, text_review",
                    intent="feature_selection",
                    columns="city, text_review",
                    method="none",
                    other_params="none",
                ).with_inputs("task", "dataset_columns", "intent"),
                dspy.Example(
                    task="select related features to columns: city",
                    dataset_columns="id, age, income, education_years, target, city, text_review",
                    intent="feature_selection",
                    columns="city",
                    method="none",
                    other_params="none",
                ).with_inputs("task", "dataset_columns", "intent"),
                # ── handle_missing_values ───────────────────────────────
                dspy.Example(
                    task="fill missing values with mean for age and income",
                    dataset_columns="age, income, education_years, Churn",
                    intent="handle_missing_values",
                    columns="age, income",
                    method="mean",
                    other_params="none",
                ).with_inputs("task", "dataset_columns", "intent"),
                # ── detect_outliers ─────────────────────────────────────
                dspy.Example(
                    task="remove outliers using IQR in income",
                    dataset_columns="age, income, education_years, Churn",
                    intent="detect_outliers",
                    columns="income",
                    method="IQR",
                    other_params="none",
                ).with_inputs("task", "dataset_columns", "intent"),
                # ── encode_categorical ──────────────────────────────────
                dspy.Example(
                    task="encode category column with one-hot",
                    dataset_columns="category, age, income, Churn",
                    intent="encode_categorical",
                    columns="category",
                    method="one-hot",
                    other_params="none",
                ).with_inputs("task", "dataset_columns", "intent"),
                # ── remove_duplicates ───────────────────────────────────
                dspy.Example(
                    task="remove duplicates by id",
                    dataset_columns="id, age, income, Churn",
                    intent="remove_duplicates",
                    columns="id",
                    method="none",
                    other_params="none",
                ).with_inputs("task", "dataset_columns", "intent"),
                # ── scale_numerical ─────────────────────────────────────
                dspy.Example(
                    task="scale age and income with standard scaler",
                    dataset_columns="age, income, education_years, Churn",
                    intent="scale_numerical",
                    columns="age, income",
                    method="standard",
                    other_params="none",
                ).with_inputs("task", "dataset_columns", "intent"),
                # ── standardize_data ────────────────────────────────────
                dspy.Example(
                    task="standardize the category column and date formats",
                    dataset_columns="name, category, date, salary",
                    intent="standardize_data",
                    columns="category, date",
                    method="none",
                    other_params="none",
                ).with_inputs("task", "dataset_columns", "intent"),
                # ── feature_engineering ─────────────────────────────────
                dspy.Example(
                    task="create new features from age and income for modeling",
                    dataset_columns="age, income, education_years, Churn",
                    intent="feature_engineering",
                    columns="age, income",
                    method="none",
                    other_params="none",
                ).with_inputs("task", "dataset_columns", "intent"),
            ]

        def forward(self, user_command, dataset_columns=""):
            # Step 0: Check original command against keywords BEFORE DSPy split_tasks.
            # For simple (non-compound) commands that match a keyword, skip the LLM call entirely.
            if "and" not in user_command.lower():
                orig_intent = self._keyword_classify(user_command)
                if orig_intent:
                    cols = self._extract_columns_from_task(user_command, dataset_columns)
                    if not cols and dataset_columns:
                        cols = [c.strip() for c in dataset_columns.split(",") if c.strip()]
                    print(f"[DEBUG] Step 0 keyword match: {user_command} → {orig_intent}")
                    return [{
                        'task': user_command,
                        'intent': orig_intent,
                        'confidence': 0.98,
                        'reasoning': f"keyword match → {orig_intent}",
                        'columns': cols,
                        'method': self._extract_method_from_text(user_command),
                        'other_params': {},
                    }]
            # Step 1: Split into tasks (DSPy, needed for compound commands)
            split_result = self.split_tasks(user_command=user_command, dataset_columns=dataset_columns)
            # Parse tasks
            tasks = [t.strip() for t in split_result.tasks.split('\n') if t.strip()]
            # Smart fallback: if DSPy didn't split, split on "and" ourselves
            if not tasks or all(user_command.lower().strip() == t.lower().strip() for t in tasks):
                if " and " in user_command.lower():
                    tasks = [t.strip() for t in user_command.split(" and ") if t.strip()]
                else:
                    tasks = [user_command]
            # Step 2: Split any tasks that still contain " and " (DSPy missed them)
            expanded = []
            for t in tasks:
                if " and " in t.lower():
                    expanded.extend([s.strip() for s in t.split(" and ") if s.strip()])
                else:
                    expanded.append(t)
            tasks = expanded
            # Step 3: Process each task
            results = []
            for task in tasks:
                # Skip empty tasks
                if not task or len(task.strip()) < 3:
                    continue
                try:
                    # Layer 1: Keyword classification first (deterministic, no DSPy call)
                    kw_intent = self._keyword_classify(task)
                    if kw_intent:
                        intent = kw_intent
                        cols = self._extract_columns_from_task(task, dataset_columns)
                        method = self._extract_method_from_text(task)
                        other_params = {}
                        confidence = 0.98
                        reasoning = f"keyword match → {kw_intent}"
                        print(f"[DEBUG] Keyword classified: {task} → {kw_intent}")
                        if cols:
                            print(f"[DEBUG] Extracted columns from task: {cols}")
                    else:
                        # Layer 2: DSPy classification
                        intent_result = self.classify(task=task)
                        intent = intent_result.intent
                        # Extract parameters
                        param_result = self.extract_params(
                            task=task,
                            dataset_columns=dataset_columns,
                            intent=intent
                        )
                        # Parse columns from model output
                        cols = self._parse_list(param_result.columns)
                        method = param_result.method if str(param_result.method).lower() != 'none' else None
                        other_params = self._parse_params(param_result.other_params)
                        confidence = self._parse_confidence(intent_result.confidence)
                        reasoning = intent_result.reasoning
                        print(f"[DEBUG] Parsed columns: {param_result.columns}")
                        print(f"[DEBUG] Parsed method: {param_result.method}")
                        print(f"[DEBUG] Parsed other_params: {param_result.other_params}")
                    # Layer 3: Post-processing corrections (applies to ALL match methods)
                    if "standardiz" in task.lower() and intent == "scale_numerical":
                        intent = "standardize_data"
                        cols = []  # Clear columns — agent scans all categoricals
                        method = None
                        print(f"[DEBUG] Corrected scale_numerical → standardize_data for: {task}")
                    elif "z-score" in task.lower() and intent == "standardize_data":
                        intent = "scale_numerical"
                        print(f"[DEBUG] Corrected standardize_data → scale_numerical for: {task} (has z-score)")
                    # If no specific columns mentioned → use all dataset columns
                    if not cols and dataset_columns:
                        cols = [c.strip() for c in dataset_columns.split(",") if c.strip()]
                    print(f"[DEBUG] intent: {intent}")
                    results.append({
                        'task': task,
                        'intent': intent,
                        'confidence': confidence,
                        'reasoning': reasoning,
                        'columns': cols,
                        'method': method,
                        'other_params': other_params,
                    })
                except Exception as e:
                    st.warning(f"⚠️ Error processing task '{task}': {str(e)}")
                    continue
            # Fallback: if no results produced but original command matches a keyword
            if not results:
                fallback_intent = self._keyword_classify(user_command)
                if fallback_intent:
                    print(f"[DEBUG] Fallback keyword match on original: {user_command} → {fallback_intent}")
                    results.append({
                        'task': user_command,
                        'intent': fallback_intent,
                        'confidence': 0.98,
                        'reasoning': f"fallback keyword match → {fallback_intent}",
                        'columns': [c.strip() for c in dataset_columns.split(",") if c.strip()] if dataset_columns else [],
                        'method': self._extract_method_from_text(user_command),
                        'other_params': {},
                    })
            return results

        def _parse_confidence(self, conf_str):
            """Parse confidence score from string"""
            try:
                match = re.search(r'0?\.\d+|\d+\.?\d*', str(conf_str))
                if match:
                    val = float(match.group())
                    # Normalize to 0-1 range
                    if val > 1:
                        val = val / 100
                    return min(max(val, 0.0), 1.0)
                return 0.85
            except:
                return 0.0

        def _parse_list(self, list_str):
            """Parse comma-separated list"""
            if str(list_str).lower() == 'none':
                return []
            items = [item.strip() for item in str(list_str).split(',') if item.strip()]
            return [item for item in items if item.lower() != 'none']

        def _parse_params(self, params_str):
            """Parse key:value parameters"""
            if str(params_str).lower() == 'none':
                return {}
            params = {}
            try:
                for pair in str(params_str).split(','):
                    if ':' in pair:
                        k, v = pair.split(':', 1)
                        params[k.strip()] = v.strip()
            except:
                pass
            return params
    class ExplainStep(dspy.Signature):
        """Ask the LM to explain why a preprocessing step ran and explain the step in detail and why this method was chosen and what is the changes that happened in the data, given metadata before/after."""
        step_name = dspy.InputField(desc="Name of the preprocessing step")
        task = dspy.InputField(desc="Original user task / intent (optional)", default="")
        metadata_before = dspy.InputField(desc="Metadata before step (JSON string)", default="")
        metadata_after = dspy.InputField(desc="Metadata after step (JSON string)", default="")
        explanation = dspy.OutputField(desc="LLM explanation for why the step was executed and what is the changes that happened in the data")

    def explain_step_llm(self,
                         step_name: str,
                         task: str = "",
                         metadata_before: Optional[Dict[str, Any]] = None,
                         metadata_after: Optional[Dict[str, Any]] = None,
                         max_tokens: int = 250) -> str:
        """
        Use litellm directly (bypassing DSPy) to produce a human-readable explanation why
        the given preprocessing step executed. Includes automatic key rotation on rate limits.
        Returns the LLM's explanation string.
        """
        from business_logic.services.retry_handler import GroqRetryHandler

        def _task():
            lm = NLPService._lm
            if lm is None:
                api_key = _key_manager.get_current_key()
                if not api_key:
                    return "Explanation failed: No API key available."
                lm = dspy.LM(model="groq/llama-3.3-70b-versatile", api_key=api_key, max_tokens=1000)
                NLPService._lm = lm
                self.lm = lm

            lm_key = lm.kwargs.get("api_key", "NOT_FOUND")
            env_key = os.environ.get("GROQ_API_KEY", "NOT_SET")
            mgr_key = _key_manager.get_current_key()
            print(f"[DIAG] explain_step_llm using lm.kwargs api_key={lm_key[:20]}...")
            print(f"[DIAG] GROQ_API_KEY env={env_key[:20]}...")
            print(f"[DIAG] key_manager current={mgr_key[:20]}...")

            mb = json.dumps(metadata_before or {}, indent=2, default=str)
            ma = json.dumps(metadata_after or {}, indent=2, default=str)
            def _truncate(s, n=2000):
                return s if len(s) <= n else (s[:n] + "\n... (truncated)")
            mb = _truncate(mb)
            ma = _truncate(ma)

            prompt = (
                f"Explain why the following data preprocessing step was applied and its impact.\n\n"
                f"Step: {step_name}\n"
                f"User request: {task or 'N/A'}\n\n"
                f"Metadata before step:\n{mb}\n\n"
                f"Metadata after step:\n{ma}\n\n"
                f"Provide a concise explanation (2-4 sentences) in natural language."
            )

            import litellm
            response = litellm.completion(
                model="groq/llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                api_key=lm_key,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return response["choices"][0]["message"]["content"].strip()

        def after_rotate(new_key):
            os.environ["GROQ_API_KEY"] = new_key
            lm = dspy.LM(model="groq/llama-3.3-70b-versatile", api_key=new_key, max_tokens=1000)
            NLPService._lm = lm
            self.lm = lm

        handler = GroqRetryHandler(_key_manager, log_fn=print)
        try:
            return handler.execute(task=_task, after_rotate=after_rotate, task_name="explain_operation")
        except RuntimeError as e:
            return f"Explanation failed: {e}"
    # @st.cache_resource
    # def build_pipeline(self, training_data: Optional[pd.DataFrame]):
    #     """Create and cache the pipeline module."""
    #     if training_data is not None and len(training_data) > 0:
    #         return AutoPrepApp.OptimizedIntentPipeline(training_examples=training_data)
    #     return AutoPrepApp.OptimizedIntentPipeline(training_examples=None)
    def run(self, user_input: str, dataset_df: Optional[pd.DataFrame] = None, dataset_path: Optional[str] = None) -> Optional[List[str]]:
        """Headless version of runUI: perform same processing without Streamlit and return detected intents.
        Includes automatic key rotation on rate limit errors.

        Args:
            user_input: command string to process (required).
            dataset_df: optional pandas DataFrame representing the dataset (preferred).
            dataset_path: optional path to CSV/Excel dataset (used if dataset_df is None).

        Returns:
            List of detected intent names (same as runUI returns on success), or [] if nothing detected.
        """
        # 1) Ensure LM is configured with key rotation capability
        max_retries = _key_manager.get_total_keys_count()
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Configure with current key
                api_key = _key_manager.get_current_key()
                os.environ["GROQ_API_KEY"] = api_key
                lm = dspy.LM(model="groq/llama-3.3-70b-versatile", api_key=api_key, max_tokens=1000)
                self.lm = lm
                NLPService._lm = lm
                
                print(f"✅ Using API Key #{_key_manager.current_index + 1}/{_key_manager.get_total_keys_count()}")
                break
                
            except Exception as e:
                error_msg = str(e).lower()
                if "rate" in error_msg or "quota" in error_msg or "limit" in error_msg or "429" in error_msg:
                    print(f"⚠️ Key #{_key_manager.current_index + 1} rate limited. Rotating...")
                    _key_manager.mark_key_failed()
                    try:
                        _key_manager.rotate_key()
                        retry_count += 1
                        time.sleep(1)
                        continue
                    except RuntimeError as rotate_err:
                        raise RuntimeError(str(rotate_err))
                else:
                    raise

        # 2) Load training data (headless: read from disk if not provided)
        # try:
        #     training_data = pd.read_csv(self.training_csv)
        # except Exception:
        #     training_data = None
        # self.training_data = training_data

        # 3) Build pipeline
        # self.pipeline = self.build_pipeline(self.training_data)

        # Use the class-level cached pipeline instead:
        self.pipeline = NLPService._pipeline 

        # 4) Load dataset columns (if a dataset provided)
        df = None
        if dataset_df is not None:
            df = dataset_df
        elif dataset_path:
            if dataset_path.lower().endswith(".csv"):
                df = pd.read_csv(dataset_path)
            else:
                df = pd.read_excel(dataset_path)

        columns_str = ", ".join(df.columns.tolist()) if df is not None else ""

        # 5) Validate input
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("user_input must be a non-empty string.")

        # 6) Pre-filter: detect obvious non-data requests without calling the LLM
        if self._is_unknown_intent(user_input):
            print(f"🔍 Pre-filter: '{user_input}' detected as unknown_intent (physical-world keywords, no data keywords)")
            intents = [["unknown_intent", "none", "none", "none"]]
            return df, intents

        # 7) Run pipeline with rate limit handling
        max_pipeline_retries = _key_manager.get_total_keys_count()
        pipeline_retry = 0
        
        while pipeline_retry < max_pipeline_retries:
            try:
                with dspy.context(lm=self.lm):
                    results = self.pipeline(user_command=user_input, dataset_columns=columns_str)
                break
                
            except Exception as e:
                error_msg = str(e).lower()
                if "rate" in error_msg or "quota" in error_msg or "limit" in error_msg or "429" in error_msg:
                    print(f"⚠️ Key #{_key_manager.current_index + 1} rate limited during processing. Rotating...")
                    _key_manager.mark_key_failed()
                    try:
                        api_key = _key_manager.rotate_key()
                        os.environ["GROQ_API_KEY"] = api_key
                        lm = dspy.LM(model="groq/llama-3.3-70b-versatile", api_key=api_key, max_tokens=1000)
                        self.lm = lm
                        NLPService._lm = lm
                        # Rebuild pipeline with new key
                        self.pipeline = self.build_pipeline()
                        pipeline_retry += 1
                        time.sleep(1)
                        continue
                    except RuntimeError as rotate_err:
                        raise RuntimeError(str(rotate_err))
                else:
                    raise RuntimeError(f"Error processing command: {e}")

        if not results:
            return []

        # 7.5) Post-process: disambiguate common LLM misclassifications
        # The LLM often maps "standardize" to scale_numerical instead of standardize_data
        user_lower = user_input.lower().strip()
        for result in results:
            intent = result.get("intent", "")
            # "standardize" without "scale"/"numeric" → standardize_data
            if intent == "scale_numerical" and "standardize" in user_lower and "scale" not in user_lower:
                result["intent"] = "standardize_data"
                result["columns"] = []  # Clear columns — agent scans all categoricals
            # "scale" without "standardize" → scale_numerical
            if intent == "standardize_data" and "scale" in user_lower and "standardize" not in user_lower:
                result["intent"] = "scale_numerical"
            # "z-score" → always scale_numerical (z-score is a numerical scaling method)
            if intent == "standardize_data" and "z-score" in user_lower:
                result["intent"] = "scale_numerical"

        # 8) Produce the same "intents" list as runUI
        intents = []
        for result in results:
            temp = []
            temp.append(result.get("intent"))
            if result.get("columns"):
                temp.append(result.get("columns"))
            if result.get("method"):
                temp.append(result.get("method"))
            if result.get("other_params"):
                temp.append(result.get("other_params"))
            intents.append(temp)
        print(intents)
        return df,intents

    @staticmethod
    def _is_unknown_intent(user_input: str) -> bool:
        """Pre-filter: detect obvious non-data requests without calling the LLM.

        Checks if the input contains physical-world action keywords AND lacks
        any data-related keywords. Returns True if it's almost certainly not
        a data preprocessing request.
        """
        text = user_input.lower().strip()

        # Physical-world action verbs (look like data terms but aren't)
        physical_verbs = {
            'clean', 'cook', 'bake', 'wash', 'paint', 'draw', 'sing', 'dance',
            'run', 'jump', 'play', 'read', 'write', 'drink', 'eat', 'sleep',
            'walk', 'talk', 'listen', 'watch', 'buy', 'sell', 'build', 'fix',
            'drive', 'swim', 'climb', 'throw', 'catch', 'cut', 'chop', 'mix',
            'stir', 'boil', 'fry', 'grill', 'roast', 'steam', 'freeze',
            'organize', 'tidy', 'scrub', 'polish', 'dust', 'mop', 'sweep', 'vacuum',
        }

        # Physical/non-data objects
        physical_objects = {
            'head', 'top', 'floor', 'room', 'house', 'car', 'kitchen', 'bathroom',
            'garage', 'garden', 'table', 'window', 'door', 'shoes', 'clothes',
            'laundry', 'dishes', 'shower', 'bath', 'hair', 'teeth', 'hands', 'face',
            'nails', 'bed', 'dinner', 'lunch', 'breakfast', 'cake', 'music', 'song',
            'movie', 'tv', 'book', 'football', 'basketball', 'tennis', 'dog', 'cat',
        }

        # Data-related keywords (if any appear, it MIGHT be data preprocessing)
        data_keywords = {
            'column', 'row', 'dataset', 'dataframe', 'csv', 'excel', 'table',
            'missing', 'null', 'nan', 'outlier', 'duplicate', 'encoding',
            'scaling', 'normalize', 'standardize', 'impute', 'fill', 'drop',
            'remove', 'spelling', 'inconsistency', 'feature', 'target', 'label',
            'category', 'numeric', 'categorical', 'text', 'string', 'date',
            'model', 'predict', 'analysis', 'data', 'value', 'values',
        }

        words = set(text.split())
        found_physical_verb = any(v in text for v in physical_verbs)
        found_physical_object = any(v in text for v in physical_objects)
        found_data_keyword = any(v in text for v in data_keywords)

        # If we see a physical verb + object combo with NO data keywords,
        # it's almost certainly not data preprocessing
        if found_physical_verb and found_physical_object and not found_data_keyword:
            return True

        return False