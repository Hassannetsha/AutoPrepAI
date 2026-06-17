import pandas as pd
from difflib import SequenceMatcher

from detector import SemanticDuplicateDetector

class Evaluator:
    def __init__(self, detector: SemanticDuplicateDetector) -> None:
        self.detector = detector

    def threshold_sensitivity_analysis(
        self,
        df: pd.DataFrame,
        text_column: str,
        ground_truth_pairs: list[tuple[int, int]],
        thresholds: list[float] | None = None
    ) -> pd.DataFrame:
        if thresholds is None:
            thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

        gt_set = {tuple(sorted(p)) for p in ground_truth_pairs}
        results = []

        for t in thresholds:
            pairs = self.detector.find_duplicates(df, text_column, threshold=t)
            
            predicted_set = self._pairs_df_to_set(pairs)

            tp = len(predicted_set & gt_set)
            fp = len(predicted_set - gt_set)
            fn = len(gt_set - predicted_set)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)

            results.append({
                "threshold": t,
                "precision": round(precision, 3),
                "recall":    round(recall, 3),
                "f1":        round(f1, 3),
                "tp": tp, "fp": fp, "fn": fn
            })
            print(f"  θ={t:.2f} → P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}  "
                  f"(TP={tp}, FP={fp}, FN={fn})")

        return pd.DataFrame(results)
    
    #  Fuzzy matching baseline  
    def fuzzy_baseline(
        self,
        df: pd.DataFrame,
        text_column: str,
        fuzzy_threshold: float = 0.90
    ) -> pd.DataFrame:
        texts = df[text_column].tolist()
        pairs = []

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                ratio = SequenceMatcher(None, texts[i], texts[j]).ratio()
                if ratio >= fuzzy_threshold:
                    pairs.append({
                        "query_index_1": i,
                        "query_index_2": j,
                        "similarity": round(ratio, 3),
                        "text_1": texts[i],
                        "text_2": texts[j]
                    })
        print(f"Fuzzy baseline (θ={fuzzy_threshold}): found {len(pairs)} duplicate pairs.")
        return pd.DataFrame(pairs)

    #  Error analysis         
    def error_analysis(
        self,
        pairs: pd.DataFrame,
        ground_truth_pairs: list[tuple[int, int]],
        df: pd.DataFrame,
        text_column: str,
        n_samples: int = 10
    ) -> dict:
        gt_set = {tuple(sorted(p)) for p in ground_truth_pairs}
        predicted_set = self._pairs_df_to_set(pairs)

        fp_pairs = predicted_set - gt_set
        fn_pairs = gt_set - predicted_set

        result = {
            "false_positives": self._pairs_to_records(fp_pairs, df, text_column, n_samples),
            "false_negatives": self._pairs_to_records(fn_pairs, df, text_column, n_samples),
            "summary": {
                "total_predicted": len(predicted_set),
                "total_ground_truth": len(gt_set),
                "false_positives_count": len(fp_pairs),
                "false_negatives_count": len(fn_pairs),
            }
        }

        print("\n=== Error Analysis Summary ===")
        for k, v in result["summary"].items():
            print(f"  {k}: {v}")

        return result
    
    #  Private helpers                                                     #
    @staticmethod
    def _pairs_df_to_set(pairs: pd.DataFrame) -> set[tuple[int, int]]:
        return {
            tuple(sorted((int(r["query_index_1"]), int(r["query_index_2"]))))
            for _, r in pairs.iterrows()
        }
 
    @staticmethod
    def _pairs_to_records(
        pair_set: set[tuple[int, int]],
        df: pd.DataFrame,
        text_column: str,
        n_samples: int,
    ) -> list[dict]:
        records = []
        for (i, j) in list(pair_set)[:n_samples]:
            records.append({
                "index_1": i,
                "index_2": j,
                "text_1":  df.iloc[i][text_column],
                "text_2":  df.iloc[j][text_column],
            })
        return records
 
