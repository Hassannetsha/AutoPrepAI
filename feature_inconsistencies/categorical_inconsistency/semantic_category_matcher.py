import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from duplicates.semantic_duplicate_remover_service import SemanticDuplicateRemover

class SemanticCategoryMatcher:
    def __init__(self, threshold=0.85):
        self.remover = SemanticDuplicateRemover(threshold=threshold, model_name="all-MiniLM-L6-v2")

    def find_similar_groups(self, values: list[str]):
        # Create a fake dataframe to reuse the FAISS search logic
        import pandas as pd
        df = pd.DataFrame({"text": values})
        _, duplicates_df = self.remover.remove_duplicates(df, "text")
        if duplicates_df.empty:
            return []

        # Group semantically close values
        groups = []
        for _, row in duplicates_df.iterrows():
            pair = sorted([row["text_1"], row["text_2"]])
            if not any(pair[0] in g or pair[1] in g for g in groups):
                groups.append(pair)
        return groups
