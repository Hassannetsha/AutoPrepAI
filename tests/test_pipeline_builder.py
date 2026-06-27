"""Tests for PipelineBuilder."""

import pandas as pd
from pipeline_builder import PipelineBuilder
from pipeline import Pipeline
from pipeline_node import PipelineNode


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

    def test_build_custom_pipeline_all_agents(self):
        agent_names = ["nlp", "datatype", "standardizer",
                       "duplicate", "outlier", "missing", "feature_engineering",
                       "feature_selection", "scaler", "encoder"]
        pipeline = PipelineBuilder.build_custom_pipeline(agent_names)
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.agents) == 10

    def test_build_custom_pipeline_subset(self):
        pipeline = PipelineBuilder.build_custom_pipeline(["missing", "scaler", "encoder"])
        assert len(pipeline.agents) == 3
        assert pipeline.agents[0].get_agent_name() == "Missing Values"
        assert pipeline.agents[1].get_agent_name() == "Scaler"
        assert pipeline.agents[2].get_agent_name() == "Encoder"

    def test_build_custom_pipeline_invalid_name(self):
        pipeline = PipelineBuilder.build_custom_pipeline(["nonexistent_agent"])
        assert len(pipeline.agents) == 0

    def test_build_custom_pipeline_mixed_valid_invalid(self):
        pipeline = PipelineBuilder.build_custom_pipeline(["missing", "invalid", "scaler"])
        assert len(pipeline.agents) == 2
        names = [a.get_agent_name() for a in pipeline.agents]
        assert "Missing Values" in names
        assert "Scaler" in names

    def test_build_custom_pipeline_nlp_always_true(self):
        pipeline = PipelineBuilder.build_custom_pipeline(["nlp"])
        ctx = DataContextEmpty()
        assert pipeline.agents[0].should_run(ctx) is True

    def test_build_default_pipeline_agent_names(self):
        pipeline = PipelineBuilder.build_default_pipeline(normalized_mode="auto")
        names = [a.get_agent_name() for a in pipeline.agents]
        assert names == self.EXPECTED_NAMES


class DataContextEmpty:
    """Minimal stand-in for DataContext with metadata only, for condition testing."""
    metadata = {}
