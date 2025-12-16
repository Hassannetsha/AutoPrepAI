# ...existing code...
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Callable, List, Tuple
import pandas as pd
from class_nlp import class_nlp
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
        return context


class MissingValueHandler(PreprocessingStep):
    name = "Missing Values"

    def run(self, context: DataContext, **kwargs) -> DataContext:
        context.log("Handling missing values")
        context.metadata["missing_values_handled"] = True
        return context


class FeatureSelector(PreprocessingStep):
    name = "Feature Selection"

    def run(self, context: DataContext, **kwargs) -> DataContext:
        context.log("Selecting features")
        context.metadata["features_selected"] = True
        return context


class Scaler(PreprocessingStep):
    name = "Scaler"

    def run(self, context: DataContext, **kwargs) -> DataContext:
        context.log("Scaling numerical features")
        context.metadata["scaled"] = True
        return context


class Encoder(PreprocessingStep):
    name = "Encoder"

    def run(self, context: DataContext, **kwargs) -> DataContext:
        context.log("Encoding categorical features")
        context.metadata["encoded"] = True
        return context


# =============================
# Pipeline Orchestrator
# =============================
class PreprocessingPipeline:
    def __init__(self, steps: List[Tuple[PreprocessingStep, Callable[[DataContext], bool]]]):
        self.steps = steps

    def run(self, context: DataContext, user_command: str = "") -> DataContext:
        """Run all steps in order. Pass user_command to steps that accept it."""
        for step, condition in self.steps:
            if condition(context):
                print(f"--> Running step: {step.name}")
                # pass user_command to every step; steps that don't use it ignore via **kwargs
                context = step.run(context, user_command=user_command)
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

def build_pipeline() -> PreprocessingPipeline:
    return PreprocessingPipeline([
        (NLPPreprocessor(), lambda ctx: ctx.metadata.get("has_text", True)),
        (InconsistencyResolver(), needs_any_intent("remove_inconsistencies")),
        (DuplicateRemover(), needs_any_intent("remove_duplicates")),
        (OutlierRemover(), needs_any_intent("keep_outliers","remove_outliers")),
        (MissingValueHandler(), needs_any_intent("handle_missing_values")),
        (FeatureSelector(), needs_any_intent("select_features")),
        (Scaler(), needs_any_intent("scale_numerical")),
        (Encoder(), needs_any_intent("encode_categorical")),
    ])
# ...existing code...
if __name__ == "__main__":
    # Pretend this is a pandas DataFrame
    raw_data = "DATASET_PLACEHOLDER"

    context = DataContext(
        data= pd.read_csv('/home/hassan-elkersh/Downloads/croky_age_salary.csv'),
        metadata={
            "has_text": True,
            "has_numeric": True,
            "has_categorical": True,
        },
    )

    pipeline = build_pipeline()
    final_context = pipeline.run(context, user_command="handle missing values using median for age and remove duplicates for salary")

    print("\nExecution Log:")
    for log in final_context.logs:
        print("-", log)
# ...existing code...