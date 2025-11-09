from category_detector import CategoryInconsistencyDetector
from category_resolver import CategoryInconsistencyResolver
from semantic_category_matcher import SemanticCategoryMatcher
from column_analyzer import ColumnAnalyzer

import pandas as pd

# Instantiate detectors
category_detector = CategoryInconsistencyDetector()
resolver = CategoryInconsistencyResolver()
semantic_matcher = SemanticCategoryMatcher(threshold=0.8)

# Analyzer
analyzer = ColumnAnalyzer(category_detector, semantic_matcher)

# Example DataFrame
# df = pd.DataFrame({
#     "country": ["USA", "usa", "United States", "U.S.A.", "Unknown"] * 200,
#     "status":["active", "inactive", "ERROR", "Unknown", "UNKNOWNNNNNNNN"] * 200
#     # "username": ["user_1", "user_2", "user_3", "user_4", "user_5"] 
#     # "review": ["I love this product!", "Great product!", "I absolutely love it!"]
# })

df = pd.read_csv("datasets/dirty_cafe_sales.csv")

for col in df.columns:
    result = analyzer.analyze_column(df, col)
    print(result)

    df, message = resolver.resolve(df, col, result)
    print(message)
    # print(df)
