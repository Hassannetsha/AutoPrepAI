from agents.pipeline_agent import PipelineAgent
from data_context import DataContext
from agent_params import AgentParams
from services.scaling_service import Scaler as DFScaler


class ScalingAgent(PipelineAgent):
    def __init__(self):
        super().__init__("Scaler")

    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        columns = params.columns or []
        context.log("Scaling numerical features")
        context.metadata["scaled"] = True
        # Use DFScaler from Class_scaler (avoid naming conflict with this class)
        try:
            method = columns[1].lower() if len(columns) > 1 and isinstance(columns[1], str) else "standard"

            # determine columns to scale (first arg may be list of columns)
            cols_to_scale = []
            if len(columns) > 0:
                first = columns[0]
                if isinstance(first, (list, tuple)):
                    cols_to_scale = list(first)
                elif isinstance(first, str) and "," in first:
                    cols_to_scale = [c.strip() for c in first.split(",") if c.strip()]
                else:
                    cols_to_scale = [c for c in columns if isinstance(c, str)]

            if not cols_to_scale:
                cols_to_scale = None

            scaler = DFScaler()
            context.data = scaler.scale(context.data, method=method, columns=cols_to_scale)
            if cols_to_scale:
                context.log(f"Scaled columns: {cols_to_scale} using method: {method}")
            else:
                context.log(f"Scaled all numeric columns using method: {method}")
        except Exception as e:
            context.log(f"Scaling error: {e}")
        return context
