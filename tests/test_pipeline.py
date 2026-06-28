"""Tests for Pipeline class."""

import pandas as pd
from unittest.mock import MagicMock, patch

from data_context import DataContext
from pipeline import Pipeline
from pipeline_node import PipelineNode
from services.nlp_service import NLPService


@patch("services.nlp_service.NLPService._init_lm")
@patch("services.nlp_service.NLPService._init_pipeline")
class TestPipeline:
    def make_mock_node(self, name="TestAgent", should_run=True, returned_ctx=None):
        node = MagicMock(spec=PipelineNode)
        node.get_agent_name.return_value = name
        node.should_run.return_value = should_run
        if returned_ctx is not None:
            node.execute.return_value = returned_ctx
        return node

    def test_init(self, mock_init_pipeline, mock_init_lm):
        agents = [self.make_mock_node()]
        pipeline = Pipeline(agents=agents)
        assert pipeline.agents == agents
        assert pipeline.session_manager is None
        assert pipeline.data_loader is None
        assert pipeline.nlp_service is not None

    def test_run_all_agents(self, mock_init_pipeline, mock_init_lm):
        ctx = DataContext(data=pd.DataFrame({"a": [1]}))
        node1 = self.make_mock_node("Agent1", returned_ctx=ctx)
        node2 = self.make_mock_node("Agent2", returned_ctx=ctx)
        pipeline = Pipeline(agents=[node1, node2])

        pipeline.nlp_service = MagicMock(spec=NLPService)
        pipeline.nlp_service.explain_step_llm.return_value = "explanation"

        result = pipeline.run(ctx, user_command="test command")

        node1.execute.assert_called_once()
        node2.execute.assert_called_once()
        assert result is ctx

    def test_run_skips_node_when_should_run_false(self, mock_init_pipeline, mock_init_lm):
        ctx = DataContext(data=pd.DataFrame({"a": [1]}))
        node1 = self.make_mock_node("SkipAgent", should_run=False)
        node2 = self.make_mock_node("RunAgent", should_run=True, returned_ctx=ctx)
        pipeline = Pipeline(agents=[node1, node2])

        pipeline.nlp_service = MagicMock(spec=NLPService)

        result = pipeline.run(ctx)

        node1.execute.assert_not_called()
        node2.execute.assert_called_once()

    def test_run_single_agent_one_step(self, mock_init_pipeline, mock_init_lm):
        ctx = DataContext(data=pd.DataFrame({"a": [1]}))
        node = self.make_mock_node("Agent1", returned_ctx=ctx)
        pipeline = Pipeline(agents=[node])
        pipeline.nlp_service = MagicMock(spec=NLPService)

        result_ctx, done = pipeline.run_single_agent(ctx, {})
        assert done is True
        node.execute.assert_called_once()

    def test_run_single_agent_no_agents(self, mock_init_pipeline, mock_init_lm):
        pipeline = Pipeline(agents=[])
        ctx = DataContext(data=pd.DataFrame({"a": [1]}))
        result_ctx, done = pipeline.run_single_agent(ctx, {})
        assert done is True
        assert result_ctx is ctx

    def test_run_single_agent_consumes_agents(self, mock_init_pipeline, mock_init_lm):
        ctx_before = DataContext(data=pd.DataFrame({"a": [1]}))
        ctx_after = DataContext(data=pd.DataFrame({"a": [2]}))  # different data = agent did work
        node1 = self.make_mock_node("Agent1", returned_ctx=ctx_after)
        node2 = self.make_mock_node("Agent2", should_run=False, returned_ctx=ctx_after)
        pipeline = Pipeline(agents=[node1, node2])
        pipeline.nlp_service = MagicMock(spec=NLPService)

        session = {}
        result_ctx, done = pipeline.run_single_agent(ctx_before, session)
        node1.execute.assert_called_once()
        node2.execute.assert_not_called()
        # agents list is no longer mutated — index tracks position in session
        assert len(pipeline.agents) == 2
        assert session.get("agent_index") == 1
        assert done is True

    def test_add_agent(self, mock_init_pipeline, mock_init_lm):
        pipeline = Pipeline(agents=[])
        node = self.make_mock_node("NewAgent")
        pipeline.add_agent(node)
        assert len(pipeline.agents) == 1
        assert pipeline.agents[0] is node

    def test_remove_agent(self, mock_init_pipeline, mock_init_lm):
        node_a = self.make_mock_node("AgentA")
        node_b = self.make_mock_node("AgentB")
        pipeline = Pipeline(agents=[node_a, node_b])
        pipeline.remove_agent("AgentA")
        assert len(pipeline.agents) == 1
        assert pipeline.agents[0].get_agent_name() == "AgentB"

    def test_remove_nonexistent_agent(self, mock_init_pipeline, mock_init_lm):
        node = self.make_mock_node("Agent")
        pipeline = Pipeline(agents=[node])
        pipeline.remove_agent("Nonexistent")
        assert len(pipeline.agents) == 1

    def test__execute_node_skips_when_should_not_run(self, mock_init_pipeline, mock_init_lm):
        node = self.make_mock_node("SkipMe", should_run=False)
        pipeline = Pipeline(agents=[node])
        ctx = DataContext(data=pd.DataFrame())

        result_ctx, executed = pipeline._execute_node(node, ctx)
        assert executed is False
        node.execute.assert_not_called()

    def test__execute_node_generates_explanation(self, mock_init_pipeline, mock_init_lm):
        ctx_before = DataContext(data=pd.DataFrame({"a": [1]}), metadata={"initial": "value"})
        ctx_after = DataContext(data=pd.DataFrame({"a": [1], "b": [2]}), metadata={"initial": "value", "new": "data"})

        node = self.make_mock_node("TestAgent", returned_ctx=ctx_after)
        pipeline = Pipeline(agents=[node])

        mock_nlp = MagicMock(spec=NLPService)
        mock_nlp.explain_step_llm.return_value = "This step was applied because..."
        pipeline.nlp_service = mock_nlp

        result_ctx, executed = pipeline._execute_node(node, ctx_before)
        assert executed is True
        mock_nlp.explain_step_llm.assert_called_once_with(
            step_name="TestAgent",
            metadata_before={"initial": "value"},
            metadata_after=mock_nlp.explain_step_llm.call_args[1]["metadata_after"]
        )
        assert result_ctx.metadata.get("explanations") is not None
        assert len(result_ctx.metadata["explanations"]) == 1
        assert result_ctx.metadata["explanations"][0]["step"] == "TestAgent"
        assert result_ctx.metadata["explanations"][0]["explanation"] == "This step was applied because..."

    def test__execute_node_nlp_agent_no_explanation(self, mock_init_pipeline, mock_init_lm):
        ctx = DataContext(data=pd.DataFrame({"a": [1]}))
        node = self.make_mock_node("NLP", returned_ctx=ctx)
        pipeline = Pipeline(agents=[node])
        mock_nlp = MagicMock(spec=NLPService)
        pipeline.nlp_service = mock_nlp

        pipeline._execute_node(node, ctx)
        mock_nlp.explain_step_llm.assert_not_called()

    def test_run_saves_execution(self, mock_init_pipeline, mock_init_lm):
        ctx = DataContext(data=pd.DataFrame({"a": [1]}))
        node = self.make_mock_node("Agent1", returned_ctx=ctx)
        session_manager = MagicMock()
        pipeline = Pipeline(agents=[node], session_manager=session_manager)
        pipeline.nlp_service = MagicMock(spec=NLPService)
        pipeline.nlp_service.explain_step_llm.return_value = "explanation"

        pipeline.run(ctx, user_command="test")
        assert ctx.metadata.get("user_command") == "test"

    def test_check_no_agents_left_to_run_all_done(self, mock_init_pipeline, mock_init_lm):
        node = self.make_mock_node("Agent1", should_run=False)
        pipeline = Pipeline(agents=[node])
        ctx = DataContext(data=pd.DataFrame())
        assert pipeline.check_no_agents_left_to_run(ctx) is True

    def test_check_no_agents_left_to_run_some_pending(self, mock_init_pipeline, mock_init_lm):
        node = self.make_mock_node("Agent1", should_run=True)
        pipeline = Pipeline(agents=[node])
        ctx = DataContext(data=pd.DataFrame())
        assert pipeline.check_no_agents_left_to_run(ctx) is False

    def test_set_nlp_service(self, mock_init_pipeline, mock_init_lm):
        pipeline = Pipeline(agents=[])
        new_service = MagicMock(spec=NLPService)
        pipeline.set_nlp_service(new_service)
        assert pipeline.nlp_service is new_service
