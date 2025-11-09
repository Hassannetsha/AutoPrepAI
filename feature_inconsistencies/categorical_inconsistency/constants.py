# constants.py

INVALID_CATEGORY_VALUES = {"unknown", "error", "n/a", "none", "null", ""}
CATEGORICAL_UNIQUE_RATIO_THRESHOLD = 0.05   # above this → not categorical
PATTERN_NON_CATEGORICAL = r".*[_\-]?\d+$"  # detects things like user_1, item-12, id_3
FUZZY_SIMILARITY_THRESHOLD = 90
