"""Tests for PipelineBuilder."""

import pandas as pd
from business_logic.cleaning_coordinator.pipeline_builder import PipelineBuilder
from business_logic.cleaning_coordinator.pipeline import Pipeline
from business_logic.cleaning_coordinator.pipeline_node import PipelineNode


class TestPipelineBuilder:
    EXPECTED_NAMES = [
        "NLP", "Data Type Inconsistency Handler",
        "Data Standardizer", "Duplicate Remover", "Outlier Remover",
        "Missing Values", "Feature Engineering",
        "Feature Selection", "Scaler", "Encoder"
    ]

    def test_build_default_pipeline_chat_mode(self):
        pipeline = PipelineBuilder.build_default_pipeline(normalized_mode="chat")
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.agents) == 10
        assert pipeline.agents[0].get_agent_name() == "NLP"
        assert pipeline.agents[1].get_agent_name() == "Data Type Inconsistency Handler"

    def test_build_default_pipeline_auto_mode(self):
        pipeline = PipelineBuilder.build_default_pipeline(normalized_mode="auto")
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.agents) == 10
        assert pipeline.agents[0].get_agent_name() == "NLP"

    def test_build_default_pipeline_nlp_condition_auto_mode(self):
        pipeline = PipelineBuilder.build_default_pipeline(normalized_mode="auto")
        ctx = DataContextEmpty()
        assert pipeline.agents[0].should_run(ctx) is False

    def test_build_default_pipeline_nlp_condition_chat_mode(self):
        pipeline = PipelineBuilder.build_default_pipeline(normalized_mode="chat")
        ctx = DataContextEmpty()
        assert pipeline.agents[0].should_run(ctx) is True

    def test_build_default_pipeline_agent_names(self):
        pipeline = PipelineBuilder.build_default_pipeline(normalized_mode="auto")
        names = [a.get_agent_name() for a in pipeline.agents]
        assert names == self.EXPECTED_NAMES


class DataContextEmpty:
    """Minimal stand-in for DataContext with metadata only, for condition testing."""
    metadata = {}
