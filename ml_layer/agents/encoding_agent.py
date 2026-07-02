from ml_layer.agents.pipeline_agent import PipelineAgent
from business_logic.cleaning_coordinator.data_context import DataContext
from business_logic.cleaning_coordinator.agent_params import AgentParams
from ml_layer.encoding.encoders.encoder_factory import EncoderFactory
from ml_layer.utils.column_detector import detect_categorical_columns


class EncodingAgent(PipelineAgent):
    def __init__(self):
        super().__init__("Encoder")

    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        context.data = context.data.reset_index(drop=True)
        context.log("Encoding categorical features")

        columns = params.columns or []

        known_methods = {"onehot", "label", "target"}
        method = params.strategy.lower() if params.strategy else "onehot"
        if len(columns) > 1 and isinstance(columns[1], str) and columns[1].lower() in known_methods:
            method = columns[1].lower()
        cols_to_encode = self._parse_columns(columns, context)
        cols_to_encode = [col for col in cols_to_encode if col in context.data.columns]

        # Skip high-cardinality columns for one-hot to prevent column explosion
        if method == "onehot":
            max_cardinality = 20
            skipped = []
            filtered = []
            for col in cols_to_encode:
                n_unique = context.data[col].nunique()
                if n_unique <= max_cardinality:
                    filtered.append(col)
                else:
                    skipped.append(f"{col} ({n_unique} unique values)")
            if skipped:
                context.log(f"Skipped high-cardinality columns: {', '.join(skipped)}")
            cols_to_encode = filtered

        if not cols_to_encode:
            context.log("No categorical columns available to encode")
            return context

        try:
            encoder = EncoderFactory.get_encoder(method)

            target = self._extract_target(columns, context)
            context.data = encoder.encode(context.data, cols_to_encode, target=target)

            context.metadata["encoded"] = True
        except Exception as e:
            print(f"Encoding error: {e}")
            context.log(f"Encoding failed.")

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

