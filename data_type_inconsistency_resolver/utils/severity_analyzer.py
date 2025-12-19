from typing import Dict, Any

def calculate_severity(result: Dict[str, Any]) -> str:
    total = result.get('total_rows', 0)
    null_count = result.get('null_count', 0)
    non_null = result.get('non_null_count', 0)


    if non_null == 0:
        return 'critical'

    detected_types = result.get('detected_types', {})
    if len(detected_types) == 0:
        return 'low'


    max_type_count = max(detected_types.values())
    max_type_pct = (max_type_count / non_null) * 100


    if max_type_pct >= 95:
        return 'low'
    elif max_type_pct >= 85:
        return 'medium'
    else:
        return 'high'


def recommend_strategy(result: Dict[str, Any], severity: str) -> str:
    recommended_type = result.get('recommended_type')


    if severity == 'low':
        return "Delete rows with inconsistent values (< 5% affected)"
    elif severity == 'medium':
        if recommended_type == 'numeric':
            return "Impute with median or mean"
        elif recommended_type == 'datetime':
            return "Delete rows or manually review"
        else:
            return "Impute with mode or delete rows"
    else:
        null_pct = (result.get('null_count', 0) / max(result.get('total_rows', 1), 1)) * 100
        if null_pct > 50:
            return "Consider deleting column (too many issues)"
        else:
            return "Manual review recommended or impute with mode"