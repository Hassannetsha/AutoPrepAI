# ...existing code...
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Callable, List, Tuple
import pandas as pd
from Class_missingValues import MissingValuesDemo
from Class_outliers import class_outliers
from class_nlp import class_nlp
from feature_selection import FeatureSelectionAgent
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


class InconsistencyResolver(PreprocessingStep):
    name = "Inconsistency Resolver"

    def run(self, context: DataContext, **kwargs) -> DataContext:
        context.log("Resolving feature inconsistencies")
        # Example: unify units, fix casing, schema issues
        context.metadata["inconsistencies_fixed"] = True
        return context


class DuplicateRemover(PreprocessingStep):
    name = "Duplicate Remover"

    def run(self, context: DataContext, **kwargs) -> DataContext:
        context.log("Removing duplicate rows")
        # Example: context.data = context.data.drop_duplicates()
        context.metadata["duplicates_removed"] = True
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
        (InconsistencyResolver(), needs_any_intent("remove_inconsistencies"),needs_any_column("remove_inconsistencies",required=False)),
        (DuplicateRemover(), needs_any_intent("remove_duplicates"),needs_any_column("remove_duplicates",required=False)),
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