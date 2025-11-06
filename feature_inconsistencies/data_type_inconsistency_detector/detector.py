import pandas as pd
from collections import Counter
from typing import Dict, Any, List
from .strategy.numeric_detector import NumericDetectionStrategy
from .strategy.datetime_detector import DatetimeDetectionStrategy
from .strategy.string_detector import StringDetectionStrategy
from .strategy.boolean_detector import BooleanDetectionStrategy
from .recommender import TypeRecommender
from .converter.numeric_converter import NumericConverter
from .converter.datetime_converter import DatetimeConverter
from .report_generator import ReportGenerator

class DataTypeInconsistencyDetector:
    def __init__(self):
        self.results = {}
        # Available detection strategies
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

    def analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        self.size = len(df)
        self.results = {}
        for col in df.columns:
            self.results[col] = self.analyze_column(df[col])
        return self.results

    def analyze_column(self, series: pd.Series) -> Dict[str, Any]:
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
        if len(non_null) == 0:
            result['recommended_type'] = 'empty_column'
            return result

        sample = non_null.sample(n=min(len(non_null), self.size), random_state=42)
        type_counts = self._detect_types(sample)
        result['detected_types'] = type_counts

        if len(type_counts) > 1:
            result['inconsistencies'].append(
                f"Multiple types detected: {', '.join([f'{k}: {v}' for k, v in type_counts.items()])}"
            )

        recommended_type = self.recommender.recommend(type_counts)
        result['recommended_type'] = recommended_type

        inconsistent_indices = []
        inconsistent_values = set()

        for idx, val in series.items():
            if pd.isna(val):
                continue
            
            # Use the SAME logic as _detect_types to classify this value
            detected_as = None
            for strategy in self.strategies:
                if strategy.detect(val):
                    detected_as = strategy.name
                    break
            
            if detected_as is None:
                detected_as = 'string'
            
            # Now check if it matches the recommended type
            if detected_as != recommended_type:
                inconsistent_indices.append(idx)
                inconsistent_values.add(val)

        result['inconsistent_indices'] = inconsistent_indices
        result['inconsistent_values'] = list(inconsistent_values)

        if result['recommended_type'] == 'numeric':
            result['conversion_issues'] = self.numeric_converter.test_conversion(non_null)
        elif result['recommended_type'] == 'datetime':
            result['conversion_issues'] = self.datetime_converter.test_conversion(non_null)

        return result

    def _detect_types(self, series: pd.Series) -> Dict[str, int]:
        type_counts = Counter()
        for val in series:
            for strategy in self.strategies:
                if strategy.detect(val):
                    type_counts[strategy.name] += 1
                    break
            else:
                type_counts['string'] += 1
        return dict(type_counts)

    def generate_report(self) -> str:
        return self.report_generator.generate(self.results)
