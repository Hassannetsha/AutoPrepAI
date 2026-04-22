import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch


class SemanticDuplicateRemoverService:
    def __init__(
        self,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        threshold: float = 0.85,
        k_neighbors: int = 10,
        batch_size: int = 512
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.k_neighbors = k_neighbors
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)
        print(f"Loaded model: {model_name}")

    # ------------------------------------------------------------------ #
    #  Core helpers                                                        #
    # ------------------------------------------------------------------ #

    def _encode(self, texts: list[str]) -> np.ndarray:
        num_workers = max(1, (os.cpu_count() or 1) - 1)
        print(f"Using {num_workers} workers for encoding.")
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                batch_size=self.batch_size
            )
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)
        return embeddings

    def _build_faiss_index(self, embeddings: np.ndarray):
        d = embeddings.shape[1]
        n_points = embeddings.shape[0]
        n_clusters = min(100, n_points)
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, n_clusters, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        return index

    def _find_duplicates(
        self,
        embeddings: np.ndarray,
        df: pd.DataFrame,
        text_column: str,
        threshold: float | None = None
    ) -> pd.DataFrame:
        t = threshold if threshold is not None else self.threshold
        index = self._build_faiss_index(embeddings)
        index.nprobe = 10
        D, I = index.search(embeddings, self.k_neighbors)

        mask = D > t
        rows, cols = np.where(mask)
        valid = rows < I[rows, cols]
        rows, cols = rows[valid], cols[valid]

        if len(rows) == 0:
            return pd.DataFrame(
                columns=["query_index_1", "query_index_2", "similarity", "text_1", "text_2"]
            )

        dupes = pd.DataFrame({
            "query_index_1": rows,
            "query_index_2": I[rows, cols],
            "similarity": D[rows, cols],
        })
        dupes["text_1"] = df.iloc[dupes["query_index_1"].values][text_column].values
        dupes["text_2"] = df.iloc[dupes["query_index_2"].values][text_column].values
        return dupes

    # ------------------------------------------------------------------ #
    #  1. Original single-column removal (unchanged)                      #
    # ------------------------------------------------------------------ #

    def remove_duplicates(
        self, df: pd.DataFrame, text_column: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        print("Encoding texts...")
        embeddings = self._encode(df[text_column].tolist())
        print("Finding semantic duplicates...")
        dupes_df = self._find_duplicates(embeddings, df, text_column)

        if dupes_df.empty:
            print("No semantic duplicates found.")
            return df, pd.DataFrame(
                columns=["query_index_1", "query_index_2", "similarity", "text_1", "text_2"]
            )

        to_remove = set(dupes_df["query_index_2"].values)
        df_dedup = df.drop(index=to_remove).reset_index(drop=True)
        print(f"Threshold {self.threshold}: found {len(to_remove)} duplicates. "
              f"{len(df)} → {len(df_dedup)} rows.")
        return df_dedup, dupes_df

    # ------------------------------------------------------------------ #
    #  2. Threshold sensitivity analysis                                  #
    # ------------------------------------------------------------------ #

    def threshold_sensitivity_analysis(
        self,
        df: pd.DataFrame,
        text_column: str,
        ground_truth_pairs: list[tuple[int, int]],
        thresholds: list[float] | None = None
    ) -> pd.DataFrame:
        """
        Sweep thresholds and report precision, recall, F1 at each level.

        Args:
            df: Input DataFrame.
            text_column: Column containing text to compare.
            ground_truth_pairs: List of (i, j) index pairs that are TRUE duplicates.
            thresholds: List of thresholds to evaluate. Defaults to 0.70–0.95.

        Returns:
            DataFrame with columns [threshold, precision, recall, f1, tp, fp, fn].
        """
        if thresholds is None:
            thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

        print("Encoding texts for sensitivity analysis...")
        embeddings = self._encode(df[text_column].tolist())
        gt_set = {tuple(sorted(p)) for p in ground_truth_pairs}

        results = []
        for t in thresholds:
            dupes_df = self._find_duplicates(embeddings, df, text_column, threshold=t)

            predicted_set = {
                tuple(sorted((int(r["query_index_1"]), int(r["query_index_2"]))))
                for _, r in dupes_df.iterrows()
            }

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

    # ------------------------------------------------------------------ #
    #  3. Multi-column entity matching                                    #
    # ------------------------------------------------------------------ #

    def _combine_columns(
        self, df: pd.DataFrame, text_columns: list[str]
    ) -> list[str]:
        """Concatenate multiple text columns with a separator."""
        combined = df[text_columns[0]].astype(str)
        for col in text_columns[1:]:
            combined = combined + " | " + df[col].astype(str)
        return combined.tolist()

    def remove_duplicates_multicolumn(
        self,
        df: pd.DataFrame,
        text_columns: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Multi-column semantic duplicate removal via text concatenation.

        Args:
            df: Input DataFrame.
            text_columns: List of column names to combine, e.g.
                          ['product_name', 'description', 'category'].

        Returns:
            (deduplicated DataFrame, duplicate-pairs DataFrame)
        """
        print(f"Combining columns for multi-column matching: {text_columns}")
        combined_texts = self._combine_columns(df, text_columns)

        embeddings = self._encode(combined_texts)

        df_temp = df.copy()
        df_temp["_combined"] = combined_texts
        dupes_df = self._find_duplicates(embeddings, df_temp, "_combined")

        if dupes_df.empty:
            print("No semantic duplicates found across combined columns.")
            return df, pd.DataFrame()

        to_remove = set(dupes_df["query_index_2"].values)
        df_dedup = df.drop(index=to_remove).reset_index(drop=True)
        print(f"Multi-column: found {len(to_remove)} duplicates. "
              f"{len(df)} → {len(df_dedup)} rows.")
        return df_dedup, dupes_df

    # ------------------------------------------------------------------ #
    #  4. Fuzzy-matching baseline for comparison                          #
    # ------------------------------------------------------------------ #

    def fuzzy_baseline(
        self,
        df: pd.DataFrame,
        text_column: str,
        fuzzy_threshold: float = 0.90
    ) -> pd.DataFrame:
        """
        Traditional fuzzy string-matching baseline (Levenshtein ratio).
        O(n²) — use only on small datasets or sampled subsets.

        Returns:
            DataFrame of detected duplicate pairs with similarity scores.
        """
        from difflib import SequenceMatcher

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

    # ------------------------------------------------------------------ #
    #  5. Error analysis                                                  #
    # ------------------------------------------------------------------ #

    def error_analysis(
        self,
        dupes_df: pd.DataFrame,
        ground_truth_pairs: list[tuple[int, int]],
        df: pd.DataFrame,
        text_column: str,
        n_samples: int = 10
    ) -> dict:
        """
        Returns false positives and false negatives for qualitative review.

        Returns:
            Dict with keys 'false_positives' and 'false_negatives',
            each a list of dicts with index pairs and their text values.
        """
        gt_set = {tuple(sorted(p)) for p in ground_truth_pairs}
        predicted_set = {
            tuple(sorted((int(r["query_index_1"]), int(r["query_index_2"]))))
            for _, r in dupes_df.iterrows()
        }

        def pairs_to_records(pair_set):
            records = []
            for (i, j) in list(pair_set)[:n_samples]:
                records.append({
                    "index_1": i, "index_2": j,
                    "text_1": df.iloc[i][text_column],
                    "text_2": df.iloc[j][text_column],
                })
            return records

        fp_pairs = predicted_set - gt_set   # predicted duplicate but was not
        fn_pairs = gt_set - predicted_set   # missed real duplicate

        result = {
            "false_positives": pairs_to_records(fp_pairs),
            "false_negatives": pairs_to_records(fn_pairs),
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