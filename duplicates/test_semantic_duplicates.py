"""
Benchmark evaluation of SemanticDuplicateRemoverService
using the WDC Product Matching Gold Standard (Computers).

Usage:
    python test_semantic_duplicates.py --gs computers_gs.json.gz
    python test_semantic_duplicates.py --gs computers_gs.json.gz --sample 500
"""

import argparse
import os
import numpy as np
import pandas as pd
from semantic_duplicate_remover_service import SemanticDuplicateRemoverService


# ------------------------------------------------------------------ #
#  Load WDC gold standard                                             #
# ------------------------------------------------------------------ #

def load_wdc_goldstandard(gs_path: str) -> pd.DataFrame:
    if os.path.isdir(gs_path):
        candidates = ["computers_gs.json.gz", "computers_gs.json"]
        resolved = None
        for name in candidates:
            candidate = os.path.join(gs_path, name)
            if os.path.exists(candidate):
                resolved = candidate
                break
        if resolved is None:
            raise FileNotFoundError(
                f"Could not find computers_gs.json(.gz) inside folder: {gs_path}"
            )
        gs_path = resolved

    print(f"Loading gold standard: {gs_path}")
    # Support both plain .json and compressed .json.gz
    compression = "gzip" if gs_path.endswith(".gz") else "infer"
    gs = pd.read_json(gs_path, lines=True, compression=compression)

    print(f"  Total pairs     : {len(gs)}")
    print(f"  Duplicate pairs : {gs['label'].sum()}")
    print(f"  Non-duplicate   : {(gs['label'] == 0).sum()}")

    gs["title_left"]  = gs["title_left"].fillna("").astype(str)
    gs["title_right"] = gs["title_right"].fillna("").astype(str)
    if "brand_left"  in gs.columns: gs["brand_left"]  = gs["brand_left"].fillna("").astype(str)
    if "brand_right" in gs.columns: gs["brand_right"] = gs["brand_right"].fillna("").astype(str)

    return gs


# ------------------------------------------------------------------ #
#  Core evaluation helpers                                            #
# ------------------------------------------------------------------ #

def _metrics(labels, predictions) -> dict:
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return dict(precision=round(precision,3), recall=round(recall,3),
                f1=round(f1,3), tp=tp, fp=fp, fn=fn, tn=tn)


# ------------------------------------------------------------------ #
#  1. Threshold sensitivity  (SBERT, title only)                     #
# ------------------------------------------------------------------ #

def sensitivity_analysis(
    gs: pd.DataFrame,
    service: SemanticDuplicateRemoverService,
    thresholds: list[float]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Encode pairs once, sweep thresholds, return results + similarities."""

    print("Encoding left titles ...")
    emb_left  = service._encode(gs["title_left"].tolist())
    print("Encoding right titles ...")
    emb_right = service._encode(gs["title_right"].tolist())

    # Cosine similarity (embeddings are already L2-normalised)
    sims   = (emb_left * emb_right).sum(axis=1)
    labels = gs["label"].values

    rows = []
    for t in thresholds:
        preds = (sims >= t).astype(int)
        m = _metrics(labels, preds)
        m["threshold"] = t
        rows.append(m)
        print(f"  θ={t:.2f} → P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  (TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")

    cols = ["threshold", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]
    return pd.DataFrame(rows)[cols], sims


# ------------------------------------------------------------------ #
#  2. Baselines                                                       #
# ------------------------------------------------------------------ #

def exact_match_baseline(gs: pd.DataFrame) -> dict:
    print("Running exact-match baseline ...")
    preds  = (gs["title_left"] == gs["title_right"]).astype(int).values
    labels = gs["label"].values
    m = _metrics(labels, preds)
    return {"method": "Exact match", **m}


def fuzzy_baseline(gs: pd.DataFrame, fuzzy_threshold: float = 0.80) -> dict:
    from difflib import SequenceMatcher
    print(f"Running fuzzy baseline (θ={fuzzy_threshold}) ...")
    labels = gs["label"].values
    preds  = np.array([
        1 if SequenceMatcher(None, r["title_left"], r["title_right"]).ratio() >= fuzzy_threshold
        else 0
        for _, r in gs.iterrows()
    ])
    m = _metrics(labels, preds)
    return {"method": f"Fuzzy (θ={fuzzy_threshold})", **m}


# ------------------------------------------------------------------ #
#  3. Multi-column  (title + brand)                                  #
# ------------------------------------------------------------------ #

def multicolumn_evaluation(
    gs: pd.DataFrame,
    service: SemanticDuplicateRemoverService,
    threshold: float,
    columns: list[str]
) -> dict:
    def combine(side: str) -> list[str]:
        parts = [gs[f"{c}_{side}"].fillna("").astype(str) for c in columns
                 if f"{c}_{side}" in gs.columns]
        combined = parts[0]
        for p in parts[1:]:
            combined = combined + " | " + p
        return combined.tolist()

    col_str = " + ".join(columns)
    print(f"\nMulti-column encoding ({col_str}) ...")
    emb_left  = service._encode(combine("left"))
    emb_right = service._encode(combine("right"))
    sims      = (emb_left * emb_right).sum(axis=1)

    preds  = (sims >= threshold).astype(int)
    labels = gs["label"].values
    m = _metrics(labels, preds)
    return {"method": f"SBERT multi-col ({col_str}) θ={threshold}", **m}


# ------------------------------------------------------------------ #
#  4. Error analysis                                                  #
# ------------------------------------------------------------------ #

def error_analysis(
    gs: pd.DataFrame,
    sims: np.ndarray,
    threshold: float,
    n_samples: int = 5
):
    labels = gs["label"].values
    preds  = (sims >= threshold).astype(int)

    fp_rows = gs[(preds == 1) & (labels == 0)]
    fn_rows = gs[(preds == 0) & (labels == 1)]

    print(f"\n--- False Positives ({len(fp_rows)} total) "
          f"— predicted duplicate but was NOT ---")
    for _, row in fp_rows.head(n_samples).iterrows():
        score = sims[row.name] if row.name < len(sims) else "?"
        print(f"  sim={score:.3f}")
        print(f"  LEFT : {str(row['title_left'])[:90]}")
        print(f"  RIGHT: {str(row['title_right'])[:90]}")
        print()

    print(f"--- False Negatives ({len(fn_rows)} total) "
          f"— missed real duplicate ---")
    for _, row in fn_rows.head(n_samples).iterrows():
        score = sims[row.name] if row.name < len(sims) else "?"
        print(f"  sim={score:.3f}")
        print(f"  LEFT : {str(row['title_left'])[:90]}")
        print(f"  RIGHT: {str(row['title_right'])[:90]}")
        print()


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs",     type=str, required=True,
                        help="Path to computers_gs.json.gz")
    parser.add_argument("--sample", type=int, default=None,
                        help="Use only first N pairs (quick smoke-test)")
    args = parser.parse_args()

    gs = load_wdc_goldstandard(args.gs)
    if args.sample:
        gs = gs.head(args.sample).reset_index(drop=True)
        print(f"Using sample of {len(gs)} pairs\n")

    service    = SemanticDuplicateRemoverService(threshold=0.85, k_neighbors=10, model_name="all-mpnet-base-v2")
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    # ── 1. Sensitivity analysis ────────────────────────────────────
    print("\n" + "=" * 60)
    print("1. THRESHOLD SENSITIVITY ANALYSIS  (title only)")
    print("=" * 60)
    sensitivity_df, sims = sensitivity_analysis(gs, service, thresholds)
    print("\nResults table:")
    print(sensitivity_df.to_string(index=False))

    best_row = sensitivity_df.loc[sensitivity_df["f1"].idxmax()]
    best_t   = float(best_row["threshold"])
    print(f"\n→ Best threshold: {best_t:.2f}  "
          f"F1={best_row['f1']:.3f}  "
          f"P={best_row['precision']:.3f}  "
          f"R={best_row['recall']:.3f}")

    # ── 2. Baselines ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. BASELINE COMPARISONS")
    print("=" * 60)
    exact = exact_match_baseline(gs)
    fuzzy = fuzzy_baseline(gs, fuzzy_threshold=0.80)
    sbert = {
        "method": f"SBERT title-only (θ={best_t})",
        **{k: best_row[k] for k in ["precision", "recall", "f1", "tp", "fp", "fn", "tn"]}
    }
    comparison = pd.DataFrame([exact, fuzzy, sbert])
    print("\n", comparison.to_string(index=False))

    # ── 3. Multi-column ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. MULTI-COLUMN  (title + brand)")
    print("=" * 60)
    mc = multicolumn_evaluation(gs, service, best_t, columns=["title", "brand"])
    print(f"  P={mc['precision']}  R={mc['recall']}  F1={mc['f1']}")

    full_comparison = pd.DataFrame([exact, fuzzy, sbert, mc])
    print("\nFull comparison (all methods):")
    print(full_comparison[["method","precision","recall","f1","tp","fp","fn"]].to_string(index=False))

    # ── 4. Error analysis ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. ERROR ANALYSIS  (best threshold)")
    print("=" * 60)
    error_analysis(gs, sims, threshold=best_t, n_samples=5)

    # ── Save CSVs ──────────────────────────────────────────────────
    sensitivity_df.to_csv("sensitivity_results.csv", index=False)
    full_comparison.to_csv("comparison_results.csv",  index=False)
    print("Saved: sensitivity_results.csv")
    print("Saved: comparison_results.csv")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # Default args for running directly in IDE
        sys.argv = ["test_semantic_duplicates.py", "--gs", "duplicates"]
    main()