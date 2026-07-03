"""
Pipeline Builder: Constructs a preprocessing pipeline with all agents configured.
"""
from typing import List

from business_logic.cleaning_coordinator.pipeline import Pipeline
from business_logic.cleaning_coordinator.pipeline_node import PipelineNode
from business_logic.cleaning_coordinator.execution_condition import AlwaysFalseCondition, IntentBasedCondition, AlwaysTrueCondition
from business_logic.cleaning_coordinator.parameter_resolver import IntentColumnResolver
from ml_layer.agents.nlp_agent import NLPAgent
from ml_layer.agents.data_type_inconsistency_agent import DataTypeInconsistencyAgent
# from ml_layer.agents.spelling_corrector_agent import SpellingCorrectorAgent
from ml_layer.agents.data_standardizing_agent import DataStandardizerAgent
from ml_layer.agents.duplicate_remover_agent import DuplicateRemoverAgent
from ml_layer.agents.outliers_agent import OutliersAgent
from ml_layer.agents.missing_value_agent import MissingValueAgent
from ml_layer.agents.feature_engineering_agent import FeatureEngineeringAgent
from ml_layer.agents.feature_selection_agent import FeatureSelectionAgent
from ml_layer.agents.scaling_agent import ScalingAgent
from ml_layer.agents.encoding_agent import EncodingAgent

class PipelineBuilder:
    """
    Builder class for constructing preprocessing pipelines.
    """
    
    @staticmethod
    def build_default_pipeline(normalized_mode:str) -> Pipeline:
        """
        Build a pipeline with all standard preprocessing agents.
        
        Returns:
            Configured Pipeline instance
        """
        nodes = []
        
        # 1. NLP Agent - Always runs first if text is available
        if normalized_mode == "chat":
            nlp_condition = AlwaysTrueCondition()
        else:
            nlp_condition = AlwaysFalseCondition()
        nlp_node = PipelineNode(
            agent=NLPAgent(),
            condition=nlp_condition,
            resolver=IntentColumnResolver([], "")
        )
        nodes.append(nlp_node)
        
        # 2. Data Type Inconsistency Handler
        datatype_node = PipelineNode(
            agent=DataTypeInconsistencyAgent(),
            condition=IntentBasedCondition(
                ["fix_data_types", "remove_inconsistencies"],
                operator="any"
            ),
            resolver=IntentColumnResolver(
                ["fix_data_types", "remove_inconsistencies"],
                ""
            )
        )
        nodes.append(datatype_node)
        
        # 4. Data Standardizer
        standardizer_node = PipelineNode(
            agent=DataStandardizerAgent(),
            condition=IntentBasedCondition(
                ["standardize_data"],
                operator="any"
            ),
            resolver=IntentColumnResolver(
                ["standardize_data"],
                ""
            )
        )
        nodes.append(standardizer_node)

        # 5. Duplicate Remover
        duplicate_node = PipelineNode(
            agent=DuplicateRemoverAgent(),
            condition=IntentBasedCondition(["remove_duplicates"], operator="any"),
            resolver=IntentColumnResolver(["remove_duplicates"], "")
        )
        nodes.append(duplicate_node)
        
        # 6. Outlier Remover
        outlier_node = PipelineNode(
            agent=OutliersAgent(),
            condition=IntentBasedCondition(
                ["remove_outliers", "detect_outliers"],
                operator="any"
            ),
            resolver=IntentColumnResolver(["remove_outliers", "detect_outliers"], "")
        )
        nodes.append(outlier_node)

        # 7. Missing Value Handler
        missing_node = PipelineNode(
            agent=MissingValueAgent(),
            condition=IntentBasedCondition(["handle_missing_values"], operator="any"),
            resolver=IntentColumnResolver(["handle_missing_values"], "mean")
        )
        nodes.append(missing_node)
        
        # 8. Feature Engineering
        feature_eng_node = PipelineNode(
            agent=FeatureEngineeringAgent(),
            condition=IntentBasedCondition(
                ["suggest_features", "feature_engineering"],
                operator="any"
            ),
            resolver=IntentColumnResolver(["suggest_features", "feature_engineering"], "")
        )
        nodes.append(feature_eng_node)
        
        # 9. Feature Selection
        feature_sel_node = PipelineNode(
            agent=FeatureSelectionAgent(),
            condition=IntentBasedCondition(
                ["select_features", "feature_selection"],
                operator="any"
            ),
            resolver=IntentColumnResolver(["select_features", "feature_selection"], "")
        )
        nodes.append(feature_sel_node)
        
        # 10. Scaler
        scaler_node = PipelineNode(
            agent=ScalingAgent(),
            condition=IntentBasedCondition(["scale_numerical"], operator="any"),
            resolver=IntentColumnResolver(["scale_numerical"], "standard")
        )
        nodes.append(scaler_node)
        
        # 11. Encoder
        encoder_node = PipelineNode(
            agent=EncodingAgent(),
            condition=IntentBasedCondition(["encode_categorical"], operator="any"),
            resolver=IntentColumnResolver(["encode_categorical"], "onehot")
        )
        nodes.append(encoder_node)
        
        # Create pipeline
        pipeline = Pipeline(
            agents=nodes,
            data_loader=None  # Can be injected later
        )
        
        return pipeline
