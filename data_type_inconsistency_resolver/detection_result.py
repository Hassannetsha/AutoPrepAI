# A thin model to document expected structure of detection results
from typing import Dict, Any


# Example usage: detection_results: Dict[str, Dict[str, Any]]
# A detection result for a column is expected to contain keys used below:
# {
# 'declared_dtype': 'object',
# 'recommended_type': 'numeric' | 'datetime' | 'string' | 'boolean',
# 'inconsistencies': True|False,
# 'conversion_issues': True|False,
# 'total_rows': int,
# 'null_count': int,
# 'non_null_count': int,
# 'detected_types': {'numeric': 10, 'string': 2, ...}
# }