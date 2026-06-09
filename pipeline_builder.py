"""
Pipeline Builder: Constructs a preprocessing pipeline with all agents configured.
"""
from typing import List

from pipeline import Pipeline
from pipeline_node import PipelineNode
from execution_condition import IntentBasedCondition, AlwaysCondition
from parameter_resolver import IntentColumnResolver
from agents.nlp_agent import NLPAgent
from agents.data_type_inconsistency_agent import DataTypeInconsistencyAgent
from agents.spelling_corrector_agent import SpellingCorrectorAgent
from agents.data_standardizing_agent import DataStandardizerAgent
from agents.duplicate_remover_agent import DuplicateRemoverAgent
from agents.outliers_agent import OutliersAgent
from agents.missing_value_agent import MissingValueAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.feature_selection_agent import FeatureSelectionAgent
from agents.scaling_agent import ScalingAgent
from agents.encoding_agent import EncodingAgent



class PipelineBuilder:
    """
    Builder class for constructing preprocessing pipelines.
    """
    
    @staticmethod
    def build_default_pipeline() -> Pipeline:
        """
        Build a pipeline with all standard preprocessing agents.
        
        Returns:
            Configured Pipeline instance
        """
        nodes = []
        
        # 1. NLP Agent - Always runs first if text is available
        nlp_node = PipelineNode(
            agent=NLPAgent(),
            condition=AlwaysCondition(),
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
        
        # 3. Spelling Corrector
        spelling_node = PipelineNode(
            agent=SpellingCorrectorAgent(),
            condition=IntentBasedCondition(
                ["correct_spelling", "remove_inconsistencies"],
                operator="any"
            ),
            resolver=IntentColumnResolver(
                ["correct_spelling", "remove_inconsistencies"],
                ""
            )
        )
        nodes.append(spelling_node)
        
        # 4. Data Standardizer
        standardizer_node = PipelineNode(
            agent=DataStandardizerAgent(),
            condition=IntentBasedCondition(
                ["standardize_data", "remove_inconsistencies"],
                operator="any"
            ),
            resolver=IntentColumnResolver(
                ["standardize_data", "remove_inconsistencies"],
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
            session_manager=None,  # Can be injected later
            data_loader=None  # Can be injected later
        )
        
        return pipeline
    
    @staticmethod
    def build_custom_pipeline(agent_list: List[str]) -> Pipeline:
        """
        Build a custom pipeline with specific agents.
        
        Args:
            agent_list: List of agent names to include
            
        Returns:
            Configured Pipeline instance with selected agents
        """
        all_agents = {
            "nlp": (NLPAgent(), ["always"], []),
            "datatype": (DataTypeInconsistencyAgent(), ["fix_data_types", "remove_inconsistencies"], ["fix_data_types", "remove_inconsistencies"]),
            "spelling": (SpellingCorrectorAgent(), ["correct_spelling", "remove_inconsistencies"], ["correct_spelling", "remove_inconsistencies"]),
            "standardizer": (DataStandardizerAgent(), ["standardize_data", "remove_inconsistencies"], ["standardize_data", "remove_inconsistencies"]),
            "duplicate": (DuplicateRemoverAgent(), ["remove_duplicates"], ["remove_duplicates"]),
            "outlier": (OutliersAgent(), ["remove_outliers", "detect_outliers"], ["remove_outliers", "detect_outliers"]),
            "missing": (MissingValueAgent(), ["handle_missing_values"], ["handle_missing_values"]),
            "feature_engineering": (FeatureEngineeringAgent(), ["suggest_features", "feature_engineering"], ["suggest_features", "feature_engineering"]),
            "feature_selection": (FeatureSelectionAgent(), ["select_features", "feature_selection"], ["select_features", "feature_selection"]),
            "scaler": (ScalingAgent(), ["scale_numerical"], ["scale_numerical"]),
            "encoder": (EncodingAgent(), ["encode_categorical"], ["encode_categorical"])
        }
        
        nodes = []
        for agent_name in agent_list:
            if agent_name.lower() in all_agents:
                agent, intents, resolver_intents = all_agents[agent_name.lower()]
                condition = AlwaysCondition() if agent_name.lower() == "nlp" else IntentBasedCondition(intents, operator="any")
                node = PipelineNode(
                    agent=agent,
                    condition=condition,
                    resolver=IntentColumnResolver(resolver_intents, "")
                )
                nodes.append(node)
        
        return Pipeline(agents=nodes, session_manager=None, data_loader=None)
