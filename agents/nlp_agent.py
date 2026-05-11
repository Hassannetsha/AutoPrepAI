from agents.pipeline_agent import PipelineAgent
from data_context import DataContext
from agent_params import AgentParams
from services.nlp_service import NLPService

class NLPAgent(PipelineAgent):
    _nlp_service = None  # class-level cache

    def __init__(self):
        super().__init__("NLP")
        # Initialize once at class level
        if NLPAgent._nlp_service is None:
            NLPAgent._nlp_service = NLPService()

    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        user_command = params.get_option("user_command", context.metadata.get("user_command", ""))
        
        if context.metadata.get("nlp_done"):
            context.log("NLP already provided/disabled; skipping NLP step")
            return context

        context.log("NLP preprocessing started")
        
        try:
            result = NLPAgent._nlp_service.run(  # ← use cached instance
                user_input=user_command, 
                dataset_df=context.data
            )
        except Exception as e:
            context.log(f"NLP error: {e}")
            return context

        df, intents = None, []
        if isinstance(result, tuple):
            if len(result) >= 2:
                df, intents = result[0], result[1]
            elif len(result) == 1:
                intents = result[0] or []
        else:
            if isinstance(result, list):
                intents = result

        if df is not None:
            context.data = df
        context.metadata["nlp_done"] = True
        context.metadata["intents"] = intents
        context.log("NLP preprocessing finished")
        return context
