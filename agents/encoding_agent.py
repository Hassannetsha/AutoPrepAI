from agents.pipeline_agent import PipelineAgent
from data_context import DataContext
from agent_params import AgentParams
from Encoders.encoder_factory import EncoderFactory
from utils.column_detector import detect_categorical_columns


class EncodingAgent(PipelineAgent):
    def __init__(self):
        super().__init__("Encoder")

    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        context.log("Encoding categorical features")

        columns = params.columns or []

        method = columns[1].lower() if len(columns) > 1 else "onehot"
        cols_to_encode = self._parse_columns(columns, context)

        try:
            encoder = EncoderFactory.get_encoder(method)

            target = self._extract_target(columns, context)
            context.data = encoder.encode(context.data, cols_to_encode, target=target)

            context.metadata["encoded"] = True
        except Exception as e:
            context.log(f"Encoding error: {e}")

        return context

    def _parse_columns(self, columns, context):
        if columns and isinstance(columns[0], (list, tuple)):
            return list(columns[0])
        if columns and isinstance(columns[0], str) and "," in columns[0]:
            return [c.strip() for c in columns[0].split(",")]
        if columns:
            return [c for c in columns if isinstance(c, str)]
        return detect_categorical_columns(context.data)

    def _extract_target(self, columns, context):
        for el in columns:
            if isinstance(el, dict) and 'target' in el:
                return el['target']
        numeric_cols = context.data.select_dtypes(include=['number']).columns.tolist()
        return numeric_cols[0] if numeric_cols else None

