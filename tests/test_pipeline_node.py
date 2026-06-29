"""Tests for PipelineNode."""

import pandas as pd
from unittest.mock import MagicMock, create_autospec

from business_logic.cleaning_coordinator.data_context import DataContext
from business_logic.cleaning_coordinator.pipeline_node import PipelineNode
from business_logic.cleaning_coordinator.execution_condition import AlwaysTrueCondition, AlwaysFalseCondition
from business_logic.cleaning_coordinator.parameter_resolver import IntentColumnResolver
from business_logic.cleaning_coordinator.agent_params import AgentParams
from ml_layer.agents.pipeline_agent import PipelineAgent


class TestPipelineNode:
    def test_should_run_true(self):
        agent = MagicMock(spec=PipelineAgent)
        condition = AlwaysTrueCondition()
        resolver = MagicMock(spec=IntentColumnResolver)
        node = PipelineNode(agent=agent, condition=condition, resolver=resolver)
        ctx = DataContext(data=pd.DataFrame())
        assert node.should_run(ctx) is True

    def test_should_run_false(self):
        agent = MagicMock(spec=PipelineAgent)
        condition = AlwaysFalseCondition()
        resolver = MagicMock(spec=IntentColumnResolver)
        node = PipelineNode(agent=agent, condition=condition, resolver=resolver)
        ctx = DataContext(data=pd.DataFrame())
        assert node.should_run(ctx) is False

    def test_resolve_params(self):
        agent = MagicMock(spec=PipelineAgent)
        condition = AlwaysTrueCondition()
        resolver = MagicMock(spec=IntentColumnResolver)
        expected_params = AgentParams(columns=["a"], strategy="mean")
        resolver.resolve.return_value = expected_params

        node = PipelineNode(agent=agent, condition=condition, resolver=resolver)
        ctx = DataContext(data=pd.DataFrame())
        result = node.resolve_params(ctx)
        assert result is expected_params
        resolver.resolve.assert_called_once_with(ctx)

    def test_execute(self):
        agent = MagicMock(spec=PipelineAgent)
        condition = AlwaysTrueCondition()
        resolver = MagicMock(spec=IntentColumnResolver)
        params = AgentParams(columns=["a"], strategy="mean")
        resolver.resolve.return_value = params
        expected_ctx = DataContext(data=pd.DataFrame({"x": [1]}))
        agent.execute.return_value = expected_ctx

        node = PipelineNode(agent=agent, condition=condition, resolver=resolver)
        ctx = DataContext(data=pd.DataFrame())
        result = node.execute(ctx)
        assert result is expected_ctx
        resolver.resolve.assert_called_once_with(ctx)
        agent.execute.assert_called_once_with(ctx, params)

    def test_get_agent_name(self):
        agent = MagicMock(spec=PipelineAgent)
        agent.name = "TestAgent"
        agent.get_agent_name.return_value = "TestAgent"

        condition = AlwaysTrueCondition()
        resolver = MagicMock(spec=IntentColumnResolver)
        node = PipelineNode(agent=agent, condition=condition, resolver=resolver)
        assert node.get_agent_name() == "TestAgent"
