from agents.pipeline_agent import PipelineAgent
from data_context import DataContext
from agent_params import AgentParams
from services.scaling_service import Scaler as DFScaler
import pandas as pd


class ScalingAgent(PipelineAgent):
    def __init__(self):
        super().__init__("Scaler")

    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        context.data = context.data.reset_index(drop=True)
        columns = params.columns or []
        context.log("Scaling numerical features")
        context.metadata["scaled"] = True
        # Use DFScaler from Class_scaler (avoid naming conflict with this class)
        try:
            known_methods = {"standard", "minmax", "robust"}
            method = params.strategy.lower() if params.strategy else "standard"
            if len(columns) > 1 and isinstance(columns[1], str) and columns[1].lower() in known_methods:
                method = columns[1].lower()

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

            numeric_cols = context.data.select_dtypes(include=['number']).columns.tolist()
            target_col = context.metadata.get("target_col")
            if cols_to_scale:
                effective_cols = [col for col in cols_to_scale if col in numeric_cols and col != target_col]
            else:
                effective_cols = [col for col in numeric_cols if col != target_col]

            scaling_fit = {
                "method": method,
                "columns": effective_cols,
                "params": {},
                "fit_scope": "train_only",
            }
            for col in effective_cols:
                values = pd.to_numeric(context.data[col], errors="coerce")
                if method == "standard":
                    mean = float(values.mean())
                    scale = float(values.std(ddof=0))
                    scaling_fit["params"][col] = {
                        "mean": mean,
                        "scale": scale if scale != 0 else 1.0,
                    }
                elif method == "minmax":
                    min_value = float(values.min())
                    max_value = float(values.max())
                    scale = max_value - min_value
                    scaling_fit["params"][col] = {
                        "min": min_value,
                        "scale": scale if scale != 0 else 1.0,
                    }
                elif method == "robust":
                    q1 = float(values.quantile(0.25))
                    median = float(values.median())
                    q3 = float(values.quantile(0.75))
                    scale = q3 - q1
                    scaling_fit["params"][col] = {
                        "median": median,
                        "scale": scale if scale != 0 else 1.0,
                    }

            scaler = DFScaler()
            context.data = scaler.scale(context.data, method=method, columns=effective_cols)
            context.metadata["scaling_fit"] = scaling_fit
            if effective_cols:
                context.log(f"Scaled columns: {effective_cols} using method: {method}")
            else:
                context.log(f"Scaled all numeric columns using method: {method}")
        except Exception as e:
            context.log(f"Scaling error: {e}")
        return context
