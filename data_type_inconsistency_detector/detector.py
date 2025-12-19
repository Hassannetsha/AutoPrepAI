import pandas as pd
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from .strategy.numeric_detector import NumericDetectionStrategy
from .strategy.datetime_detector import DatetimeDetectionStrategy
from .strategy.string_detector import StringDetectionStrategy
from .strategy.boolean_detector import BooleanDetectionStrategy
from .recommender import TypeRecommender
from .converter.numeric_converter import NumericConverter
from .converter.datetime_converter import DatetimeConverter
from .report_generator import ReportGenerator

class DataTypeInconsistencyDetector:
    def __init__(self, max_workers: int = 4):
        self.results = {}
        self.strategies = [
            NumericDetectionStrategy(),
            DatetimeDetectionStrategy(),
            BooleanDetectionStrategy(),
            StringDetectionStrategy()
        ]
        self.recommender = TypeRecommender()
        self.numeric_converter = NumericConverter()
        self.datetime_converter = DatetimeConverter()
        self.report_generator = ReportGenerator()
        self.max_workers = max_workers

    def analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        self.results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self._analyze_column_parallel, [df[col] for col in df.columns])
        self.results = {res["column_name"]: res for res in results}
        return self.results

    def _analyze_column_parallel(self, series: pd.Series) -> Dict[str, Any]:
        result = {
            'column_name': series.name,
            'total_rows': len(series),
            'null_count': series.isna().sum(),
            'non_null_count': series.notna().sum(),
            'declared_dtype': str(series.dtype),
            'detected_types': {},
            'inconsistencies': [],
            'inconsistent_indices': [],
            'inconsistent_values': [],
            'recommended_type': None,
            'conversion_issues': []
        }

        non_null = series.dropna()
        if non_null.empty:
            result['recommended_type'] = 'empty_column'
            return result

        # Detect types once for all values
        detected_types = []
        for val in non_null:
            detected_as = None
            for strategy in self.strategies:
                if strategy.detect(val):
                    detected_as = strategy.name
                    break
            if detected_as is None:
                detected_as = 'string'
            detected_types.append(detected_as)

        type_counts = dict(Counter(detected_types))
        result['detected_types'] = type_counts

        # Check for inconsistencies
        if len(type_counts) > 1:
            result['inconsistencies'].append(
                f"Multiple types detected: {', '.join([f'{k}: {v}' for k, v in type_counts.items()])}"
            )

        recommended_type = self.recommender.recommend(type_counts)
        result['recommended_type'] = recommended_type

        # Inconsistencies (avoid re-detecting)
        inconsistent_mask = [t != recommended_type for t in detected_types]
        inconsistent_indices = non_null.index[inconsistent_mask].tolist()
        inconsistent_values = non_null[inconsistent_mask].unique().tolist()

        result['inconsistent_indices'] = inconsistent_indices
        result['inconsistent_values'] = inconsistent_values

        # Conversion checks
        if recommended_type == 'numeric':
            result['conversion_issues'] = self.numeric_converter.test_conversion(non_null)
        elif recommended_type == 'datetime':
            result['conversion_issues'] = self.datetime_converter.test_conversion(non_null)

        return result

    def generate_report(self) -> str:
        return self.report_generator.generate(self.results)
