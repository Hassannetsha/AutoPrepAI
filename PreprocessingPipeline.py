# ...existing code...
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Callable, List, Tuple
import pandas as pd
from Class_missingValues import MissingValuesDemo
from Class_outliers import class_outliers
from class_nlp import class_nlp
from feature_selection import FeatureSelectionAgent
from data_type_inconsistency_detector.detector import DataTypeInconsistencyDetector
from data_type_inconsistency_resolver.resolver import DataResolver
from SpellingCorrector import SpellingCorrector
from data_standardization.feature_standardizer import FeatureStandardizer
from data_standardization.config import API_KEY
from groq import Groq
from duplicates.exact_duplicate_remover import ExactDuplicateRemover
from duplicates.semantic_duplicate_remover import SemanticDuplicateRemover
# ...existing code...

@dataclass
class DataContext:
    # ...existing code...
    data: Any  # usually pandas.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)

    def log(self, message: str):
        self.logs.append(message)


# =============================
# Base Step Interface
# =============================
class PreprocessingStep(ABC):
    name: str

    @abstractmethod
    def run(self, context: DataContext, **kwargs) -> DataContext:
        """Apply the preprocessing step and return updated context."""
        pass


# =============================
# Concrete Steps
# =============================
class NLPPreprocessor(PreprocessingStep):
    name = "NLP"

    def run(self, context: DataContext, user_command: str = "") -> DataContext:
        context.log("NLP preprocessing started")
        auto = class_nlp()
        # class_nlp.run returns (df, intents) in headless mode
        try:
            result = auto.run(user_input=user_command, dataset_df=context.data)
        except Exception as e:
            context.log(f"NLP error: {e}")
            return context

        # handle result being tuple (df, intents) or just intents
        df, intents = None, []
        if isinstance(result, tuple):
            if len(result) >= 2:
                df, intents = result[0], result[1]
            elif len(result) == 1:
                intents = result[0] or []
        else:
            # fallback: maybe returns list of intents
            if isinstance(result, list):
                intents = result
        if df is not None:
            context.data = df
        context.metadata["nlp_done"] = True
        context.metadata["intents"] = intents
        context.log("NLP preprocessing finished")
        return context


class DataTypeInconsistencyHandler(PreprocessingStep):
    name = "Data Type Inconsistency Handler"

    def run(self, context: DataContext, **kwargs) -> DataContext:
        columns = kwargs.get("columns", [])
        context.log("Detecting and resolving data type inconsistencies")
        
        try:
            # Step 1: Detect inconsistencies
            detector = DataTypeInconsistencyDetector()
            detection_results = detector.analyze_dataframe(context.data)
            
            # Store detection results in metadata
            context.metadata["datatype_detection_results"] = detection_results
            
            # Step 2: Identify columns with inconsistencies
            inconsistent_columns = []
            for col_name, result in detection_results.items():
                if len(result.get('detected_types', {})) > 1:
                    inconsistent_columns.append(col_name)
                    context.log(f"Column '{col_name}': {result['detected_types']}")
            
            if not inconsistent_columns:
                context.log("No data type inconsistencies detected")
                context.metadata["datatype_inconsistencies_fixed"] = True
                return context
            
            # Step 3: Resolve inconsistencies
            resolver = DataResolver(context.data, detection_results)
            
            for col_name in inconsistent_columns:
                result = detection_results[col_name]
                recommended_type = result.get('recommended_type')
                
                # Apply conversion strategy based on recommended type
                if recommended_type and recommended_type != 'empty_column':
                    context.log(f"Converting '{col_name}' to {recommended_type}")
                    resolver.resolve(
                        strategy_name="convert_to_type",
                        column_name=col_name,
                        target_type=recommended_type
                    )
            
            # Update context with resolved data
            context.data = resolver.df_resolved
            context.metadata["datatype_inconsistencies_fixed"] = True
            context.metadata["resolution_log"] = resolver.resolution_log
            context.log(f"Fixed data type inconsistencies in {len(inconsistent_columns)} columns")
            
        except Exception as e:
            context.log(f"Data type inconsistency handling error: {e}")
            import traceback
            context.log(traceback.format_exc())
        
        return context


class SpellingCorrectorStep(PreprocessingStep):
    name = "Spelling Corrector"

    def run(self, context: DataContext, **kwargs) -> DataContext:
        columns = kwargs.get("columns", [])
        context.log("Correcting spelling errors in categorical columns")
        
        try:
            # Identify categorical columns
            categorical_cols = context.data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not categorical_cols:
                context.log("No categorical columns found for spelling correction")
                return context
            
            # If specific columns are provided, use them; otherwise process all categorical columns
            target_cols = columns if columns else categorical_cols
            target_cols = [col for col in target_cols if col in categorical_cols]
            
            if not target_cols:
                context.log("No valid categorical columns to process")
                return context
            
            corrector = SpellingCorrector(max_edit_distance=2, prefix_length=7)
            corrected_columns = []
            
            for col in target_cols:
                try:
                    # Build dictionary from the column itself
                    corrector.build_dictionary_from_dataframe(
                        context.data, 
                        col, 
                        show_progress=False
                    )
                    
                    # Correct the column
                    context.data[col] = corrector.correct_dataframe_column(
                        context.data,
                        col,
                        show_progress=False,
                        inplace=False
                    )
                    
                    corrected_columns.append(col)
                    context.log(f"Corrected spelling in column '{col}'")
                    
                except Exception as col_error:
                    context.log(f"Error correcting column '{col}': {col_error}")
            
            context.metadata["spelling_corrected"] = True
            context.metadata["spelling_corrected_columns"] = corrected_columns
            context.log(f"Spelling correction completed for {len(corrected_columns)} columns")
            
        except Exception as e:
            context.log(f"Spelling correction error: {e}")
            import traceback
            context.log(traceback.format_exc())
        
        return context


class DataStandardizer(PreprocessingStep):
    name = "Data Standardizer"

    def run(self, context: DataContext, **kwargs) -> DataContext:
        columns = kwargs.get("columns", [])
        context.log("Standardizing data values using LLM")
        
        try:
            # Initialize Groq client
            client = Groq(api_key=API_KEY)
            model = "llama-3.3-70b-versatile"
            
            standardizer = FeatureStandardizer(context.data, client, model)
            
            # Run detection on all columns or specified columns
            if columns:
                # Process only specified columns
                for col in columns:
                    if col in context.data.columns:
                        if pd.api.types.is_numeric_dtype(context.data[col]):
                            standardizer.detect_numeric_issues(col)
                        else:
                            standardizer.detect_categorical_issues(col)
            else:
                # Process all columns
                standardizer.run_detection()
            
            # Apply categorical fixes
            standardizer.apply_categorical_fixes(columns=columns if columns else None)
            
            # Update context with standardized data
            context.data = standardizer.df
            context.metadata["data_standardized"] = True
            context.metadata["standardization_results"] = standardizer.results
            context.log("Data standardization completed")
            
        except Exception as e:
            context.log(f"Data standardization error: {e}")
            import traceback
            context.log(traceback.format_exc())
        
        return context


class ExactDuplicateRemoverStep(PreprocessingStep):
    name = "Exact Duplicate Remover"

    def run(self, context: DataContext, **kwargs) -> DataContext:
        columns = kwargs.get("columns", [])
        context.log("Removing exact duplicate rows")
        
        try:
            # Initialize exact duplicate remover
            remover = ExactDuplicateRemover(
                subset=columns if columns else None,
                keep='first',
                auto_exclude_ids=True
            )
            
            # Remove duplicates
            df_dedup, df_duplicates = remover.remove_duplicates(
                context.data,
                verbose=False
            )
            
            num_duplicates = len(df_duplicates)
            context.data = df_dedup
            context.metadata["exact_duplicates_removed"] = True
            context.metadata["exact_duplicates_count"] = num_duplicates
            
            if num_duplicates > 0:
                context.log(f"Removed {num_duplicates} exact duplicate rows")
            else:
                context.log("No exact duplicates found")
                
        except Exception as e:
            context.log(f"Exact duplicate removal error: {e}")
            import traceback
            context.log(traceback.format_exc())
        
        return context


class SemanticDuplicateRemoverStep(PreprocessingStep):
    name = "Semantic Duplicate Remover"

    def run(self, context: DataContext, **kwargs) -> DataContext:
        columns = kwargs.get("columns", [])
        context.log("Removing semantic duplicate rows")
        
        try:
            # Check if data is suitable for semantic duplicate detection
            text_columns = context.data.select_dtypes(include=['object']).columns.tolist()
            
            if not text_columns:
                context.log("No text columns found for semantic duplicate detection")
                return context
            
            # Determine which text column to use
            # Priority: specified column > column with longest average text > first text column
            target_column = None
            
            if columns and len(columns) > 0:
                # Use specified column if it's a text column
                for col in columns:
                    if col in text_columns:
                        target_column = col
                        break
            
            if not target_column:
                # Find the text column with longest average text length
                avg_lengths = {}
                for col in text_columns:
                    avg_lengths[col] = context.data[col].astype(str).str.len().mean()
                
                # Only use semantic detection if average text length is substantial (> 20 chars)
                max_col = max(avg_lengths, key=avg_lengths.get)
                if avg_lengths[max_col] > 20:
                    target_column = max_col
                else:
                    context.log(f"Text columns have short content (avg < 20 chars). Skipping semantic duplicate detection.")
                    return context
            
            if not target_column:
                context.log("No suitable text column found for semantic duplicate detection")
                return context
            
            context.log(f"Using column '{target_column}' for semantic duplicate detection")
            
            # Initialize semantic duplicate remover
            remover = SemanticDuplicateRemover(
                model_name="paraphrase-MiniLM-L6-v2",
                threshold=0.85,
                k_neighbors=10,
                batch_size=512
            )
            
            # Remove semantic duplicates
            df_dedup, df_duplicates = remover.remove_duplicates(
                context.data,
                text_column=target_column
            )
            
            num_duplicates = len(df_duplicates) if not df_duplicates.empty else 0
            context.data = df_dedup
            context.metadata["semantic_duplicates_removed"] = True
            context.metadata["semantic_duplicates_count"] = num_duplicates
            context.metadata["semantic_column_used"] = target_column
            
            if num_duplicates > 0:
                context.log(f"Removed {num_duplicates} semantic duplicate rows from column '{target_column}'")
            else:
                context.log(f"No semantic duplicates found in column '{target_column}'")
                
        except Exception as e:
            context.log(f"Semantic duplicate removal error: {e}")
            import traceback
            context.log(traceback.format_exc())
        
        return context


class OutlierRemover(PreprocessingStep):
    name = "Outlier Remover"

    def run(self, context: DataContext, **kwargs) -> DataContext:
        context.log("Handling outliers")
        context.metadata["outliers_handled"] = True
        c = class_outliers(dataframe=context.data)
        c.load()
        c.run_isolation_forest()
        cleaned = c.get_cleaned()
        context.data = cleaned
        return context


class MissingValueHandler(PreprocessingStep):
    name = "Missing Values"

    def run(self, context: DataContext, **kwargs) -> DataContext:
        columns = kwargs.get("columns", [])
        context.log("Handling missing values")
        context.metadata["missing_values_handled"] = True
        print("Handling missing values for columns:", columns)
        demo = MissingValuesDemo()
        context.data = demo.run(context.data, strategy=columns[1] if len(columns) > 1 else "Mean")
        return context


class FeatureSelector(PreprocessingStep):
    name = "Feature Selection"

    def run(self, context: DataContext, **kwargs) -> DataContext:
        columns = kwargs.get("columns", []) or []
        context.log("Selecting features")
        try:
            agent = FeatureSelectionAgent(random_state=context.metadata.get("random_state", 42))
            selected, pruned_df = agent.run(context.data, columns=columns, metadata=context.metadata, threshold=kwargs.get("threshold"))
        except ValueError as e:
            context.log(f"Feature selection skipped: {e}")
            return context
        except Exception as e:
            context.log(f"Feature selection error: {e}")
            return context
        context.data = pruned_df
        context.metadata["features_selected"] = True
        context.metadata["selected_features"] = selected
        context.log(f"Selected features: {selected}")
        return context


class Scaler(PreprocessingStep):
    name = "Scaler"

    def run(self, context: DataContext, **kwargs) -> DataContext:
        columns = kwargs.get("columns", [])
        context.log("Scaling numerical features")
        context.metadata["scaled"] = True
        scaler = Scaler()
        context.data = scaler.scale(context.data, method=columns[1] if len(columns) > 1 else "standard")
        return context


class Encoder(PreprocessingStep):
    name = "Encoder"

    def run(self, context: DataContext, **kwargs) -> DataContext:
        columns = kwargs.get("columns", [])
        context.log("Encoding categorical features")
        context.metadata["encoded"] = True
        encoder = Encoder()
        context.data = encoder.encode(context.data, method=columns[1] if len(columns) > 1 else "onehot")
        return context


# =============================
# Pipeline Orchestrator
# =============================
class PreprocessingPipeline:
    def __init__(self, steps: List[Tuple[PreprocessingStep, Callable[[DataContext], bool]]]):
        self.steps = steps

    def run(self, context: DataContext, user_command: str = "") -> DataContext:
        """Run all steps in order. Pass user_command to steps that accept it.
        Each step entry may be:
          - (step, condition)
          - (step, condition, columns_getter)
        """
        for entry in self.steps:
            # unpack flexible tuple
            if len(entry) == 2:
                step, condition = entry
                columns_getter = None
            elif len(entry) == 3:
                step, condition, columns_getter = entry
            else:
                raise ValueError(f"Invalid pipeline step tuple length: {len(entry)}")

            if condition(context):
                print(f"--> Running step: {step.name}")
                # prepare kwargs for step.run
                kwargs = {"user_command": user_command}
                if columns_getter is not None:
                    cols = columns_getter(context)
                    # always pass columns (could be empty list)
                    kwargs["columns"] = cols
                context = step.run(context, **kwargs)
            else:
                print(f"--> Skipping step: {step.name}")
        return context
def needs_any_intent(*intent_names: str) -> Callable[[DataContext], bool]:
    def checker(ctx: DataContext) -> bool:
        # if NLP didn't run, fall back to metadata capability flags
        if ctx.metadata.get("nlp_done"):
            intents = ctx.metadata.get("intents", []) or []
            # intents items may be like ["intent_name", "column_name"] or just "intent_name"
            return any(
                ((item[0] if isinstance(item, (list, tuple)) and len(item) > 0 else item) in intent_names)
                for item in intents
            )
        # fallback: allow steps if dataset has appropriate types and no NLP intent provided
        if "handle_missing_values" in intent_names and ctx.metadata.get("has_numeric", False):
            return True
        return False
    return checker
def needs_any_column(*intent_names: str,required = True) -> Callable[[DataContext], bool]:
    def checker(ctx: DataContext) -> List[str]:
        if not required:
            return []
        # if NLP didn't run, fall back to metadata capability flags
        if ctx.metadata.get("nlp_done"):
            intents = ctx.metadata.get("intents", []) or []
            # intents items may be like ["intent_name", "column_name"] or just "intent_name"
            for item in intents:
                intent_name = item[0] if isinstance(item, (list, tuple)) and len(item) > 0 else item
                if intent_name in intent_names:
                    if len(item) > 1:
                        return item[1:]
    return checker

def build_pipeline() -> PreprocessingPipeline:
    return PreprocessingPipeline([
        (NLPPreprocessor(), lambda ctx: ctx.metadata.get("has_text", True)),
        (DataTypeInconsistencyHandler(), needs_any_intent("fix_data_types", "remove_inconsistencies"),needs_any_column("fix_data_types", "remove_inconsistencies",required=False)),
        (SpellingCorrectorStep(), needs_any_intent("correct_spelling"),needs_any_column("correct_spelling",required=False)),
        (DataStandardizer(), needs_any_intent("standardize_data"),needs_any_column("standardize_data",required=False)),
        (ExactDuplicateRemoverStep(), needs_any_intent("remove_duplicates", "remove_exact_duplicates"),needs_any_column("remove_duplicates", "remove_exact_duplicates",required=False)),
        (SemanticDuplicateRemoverStep(), needs_any_intent("remove_semantic_duplicates"),needs_any_column("remove_semantic_duplicates",required=False)),
        (OutlierRemover(), needs_any_intent("remove_outliers"),needs_any_column("remove_outliers",required=False)),
        (MissingValueHandler(), needs_any_intent("handle_missing_values"),needs_any_column("handle_missing_values",required=True)),
        (FeatureSelector(), needs_any_intent("select_features"),needs_any_column("select_features",required=True)),
        (Scaler(), needs_any_intent("scale_numerical"),needs_any_column("scale_numerical",required=True)),
        (Encoder(), needs_any_intent("encode_categorical"),needs_any_column("encode_categorical",required=True)),
    ])
# ...existing code...
if __name__ == "__main__":
    # Pretend this is a pandas DataFrame
    raw_data = "DATASET_PLACEHOLDER"

    context = DataContext(
        data= pd.read_csv('/home/hassan-elkersh/graduation project/AutoPrepAI/Input/placement.csv'),
        metadata={
            "has_text": True,
            "has_numeric": True,
            "has_categorical": True,
        },
    )

    pipeline = build_pipeline()
    final_context = pipeline.run(context, user_command="handle missing values using median for cgpa and remove duplicates for placed")

    print("\nExecution Log:")
    for log in final_context.logs:
        print("-", log)
# ...existing code...