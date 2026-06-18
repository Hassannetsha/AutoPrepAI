"""
Benchmark evaluation of SemanticDuplicateRemoverService
using WDC product-matching pair datasets.

Usage:
    python test_semantic_duplicates.py
    python test_semantic_duplicates.py --gs duplicates/computers_gs.json duplicates/watches_gs.json.gz duplicates/cameras_gs.json.gz duplicates/shoes_gs.json.gz
    python test_semantic_duplicates.py --sample 500
    python test_semantic_duplicates.py --cross-encoder
"""

import argparse
import os
import re
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
    compression = "gzip" if gs_path.endswith(".gz") else "infer"
    gs = pd.read_json(gs_path, lines=True, compression=compression)
    gs["label"] = pd.to_numeric(gs["label"], errors="coerce").fillna(0).astype(int)

    print(f"  Total pairs     : {len(gs)}")
    print(f"  Duplicate pairs : {gs['label'].sum()}")
    print(f"  Non-duplicate   : {(gs['label'] == 0).sum()}")

    gs["title_left"]  = gs["title_left"].fillna("").astype(str)
    gs["title_right"] = gs["title_right"].fillna("").astype(str)
    if "brand_left"  in gs.columns: gs["brand_left"]  = gs["brand_left"].fillna("").astype(str)
    if "brand_right" in gs.columns: gs["brand_right"] = gs["brand_right"].fillna("").astype(str)

    return gs


def load_pair_dataset(path: str) -> pd.DataFrame:
    if os.path.isdir(path) or path.endswith((".json", ".json.gz")):
        return load_wdc_goldstandard(path)
    raise ValueError(f"Unsupported WDC dataset format: {path}")


def _dataset_slug(path: str) -> str:
    name = os.path.basename(os.path.normpath(path))
    if name in {"duplicates", ""}:
        name = "wdc_gold"
    name = re.sub(r"\.(json\.gz|json)$", "", name, flags=re.IGNORECASE)
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "dataset"


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
    return dict(precision=round(precision, 3), recall=round(recall, 3),
                f1=round(f1, 3), tp=tp, fp=fp, fn=fn, tn=tn)


def encode_paired_texts(
    left_texts: list[str],
    right_texts: list[str],
    service: SemanticDuplicateRemoverService,
) -> tuple[np.ndarray, np.ndarray]:
    all_texts = pd.Series(left_texts + right_texts, dtype="string").fillna("").astype(str)
    unique_texts = all_texts.drop_duplicates().tolist()
    print(f"Encoding {len(unique_texts)} unique texts for {len(left_texts)} pairs ...")

    embeddings = service.detector.encoder.encode(unique_texts)
    embedding_by_text = dict(zip(unique_texts, embeddings))
    emb_left  = np.vstack([embedding_by_text[text] for text in left_texts])
    emb_right = np.vstack([embedding_by_text[text] for text in right_texts])
    return emb_left, emb_right


# ------------------------------------------------------------------ #
#  1. Threshold sensitivity  (bi-encoder, title only)                #
# ------------------------------------------------------------------ #

def sensitivity_analysis(
    gs: pd.DataFrame,
    service: SemanticDuplicateRemoverService,
    thresholds: list[float],
) -> tuple[pd.DataFrame, np.ndarray]:
    emb_left, emb_right = encode_paired_texts(
        gs["title_left"].tolist(),
        gs["title_right"].tolist(),
        service,
    )

    sims   = (emb_left * emb_right).sum(axis=1)
    labels = gs["label"].values
    rows   = []

    for t in thresholds:
        preds = (sims >= t).astype(int)
        m = _metrics(labels, preds)
        m["threshold"] = t
        rows.append(m)
        print(f"  θ={t:.2f} │ P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
              f"(TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")

    cols = ["threshold", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]
    return pd.DataFrame(rows)[cols], sims


# ------------------------------------------------------------------ #
#  2. Cross-encoder reranking evaluation                             #
# ------------------------------------------------------------------ #

def cross_encoder_evaluation(
    gs: pd.DataFrame,
    service: SemanticDuplicateRemoverService,
    bi_encoder_threshold: float,
) -> dict:
    """
    Run the cross-encoder over bi-encoder candidates and evaluate.
    Only meaningful when service was built with use_cross_encoder=True.
    """
    if service.detector.cross_encoder is None:
        print("  Cross-encoder not enabled — skipping.")
        return {}

    sims_left, sims_right = encode_paired_texts(
        gs["title_left"].tolist(),
        gs["title_right"].tolist(),
        service,
    )
    sims = (sims_left * sims_right).sum(axis=1)
    labels = gs["label"].values

    # Only rerank pairs that passed the bi-encoder threshold
    candidate_mask = sims >= bi_encoder_threshold
    candidate_indices = np.where(candidate_mask)[0]

    if len(candidate_indices) == 0:
        print("  No candidates above bi-encoder threshold.")
        return {}

    candidate_pairs = [
        (gs["title_left"].iloc[i], gs["title_right"].iloc[i])
        for i in candidate_indices
    ]

    scores = service.detector.cross_encoder.rerank(candidate_pairs)
    ce_threshold = service.detector.cross_encoder.threshold

    # Build final predictions: candidate must pass both thresholds
    preds = np.zeros(len(gs), dtype=int)
    for idx, score, i in zip(range(len(candidate_indices)), scores, candidate_indices):
        if score >= ce_threshold:
            preds[i] = 1

    m = _metrics(labels, preds)
    print(f"  Cross-encoder (bi θ={bi_encoder_threshold}, ce θ={ce_threshold}): "
          f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
          f"(TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")
    return {"method": f"SBERT+CrossEncoder (θ={bi_encoder_threshold})", **m}


# ------------------------------------------------------------------ #
#  3. Baselines                                                       #
# ------------------------------------------------------------------ #

def exact_match_baseline(gs: pd.DataFrame) -> dict:
    print("Running exact-match baseline ...")
    preds  = (gs["title_left"] == gs["title_right"]).astype(int).values
    labels = gs["label"].values
    return {"method": "Exact match", **_metrics(labels, preds)}


def fuzzy_baseline(gs: pd.DataFrame, fuzzy_threshold: float = 0.80) -> dict:
    from difflib import SequenceMatcher
    print(f"Running fuzzy baseline (θ={fuzzy_threshold}) ...")
    labels = gs["label"].values
    preds  = np.array([
        1 if SequenceMatcher(None, r["title_left"], r["title_right"]).ratio() >= fuzzy_threshold
        else 0
        for _, r in gs.iterrows()
    ])
    return {"method": f"Fuzzy (θ={fuzzy_threshold})", **_metrics(labels, preds)}


# ------------------------------------------------------------------ #
#  4. Multi-column  (title + brand)                                  #
# ------------------------------------------------------------------ #

def multicolumn_evaluation(
    gs: pd.DataFrame,
    service: SemanticDuplicateRemoverService,
    threshold: float,
    columns: list[str],
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

    emb_left, emb_right = encode_paired_texts(combine("left"), combine("right"), service)
    sims   = (emb_left * emb_right).sum(axis=1)
    labels = gs["label"].values
    preds  = (sims >= threshold).astype(int)
    m      = _metrics(labels, preds)
    return {"method": f"SBERT multi-col ({col_str}) θ={threshold}", **m}


# ------------------------------------------------------------------ #
#  5. Error analysis                                                  #
# ------------------------------------------------------------------ #

def error_analysis(
    gs: pd.DataFrame,
    sims: np.ndarray,
    threshold: float,
    n_samples: int = 5,
    label: str = "",
):
    labels  = gs["label"].values
    preds   = (sims >= threshold).astype(int)
    fp_rows = gs[(preds == 1) & (labels == 0)]
    fn_rows = gs[(preds == 0) & (labels == 1)]

    header = f" [{label}]" if label else ""
    print(f"\n--- False Positives{header} ({len(fp_rows)} total) "
          f"— predicted duplicate but was NOT ---")
    for _, row in fp_rows.head(n_samples).iterrows():
        score = sims[row.name] if row.name < len(sims) else "?"
        print(f"  sim={score:.3f}")
        print(f"  LEFT : {str(row['title_left'])[:90]}")
        print(f"  RIGHT: {str(row['title_right'])[:90]}")
        print()

    print(f"--- False Negatives{header} ({len(fn_rows)} total) "
          f"— missed real duplicate ---")
    for _, row in fn_rows.head(n_samples).iterrows():
        score = sims[row.name] if row.name < len(sims) else "?"
        print(f"  sim={score:.3f}")
        print(f"  LEFT : {str(row['title_left'])[:90]}")
        print(f"  RIGHT: {str(row['title_right'])[:90]}")
        print()


# ------------------------------------------------------------------ #
#  Main per-dataset evaluation                                        #
# ------------------------------------------------------------------ #

def evaluate_dataset(
    dataset_path: str,
    service: SemanticDuplicateRemoverService,
    thresholds: list[float],
    sample: int | None = None,
) -> bool:
    print("\n" + "#" * 72)
    print(f"DATASET: {dataset_path}")
    print("#" * 72)

    try:
        gs = load_pair_dataset(dataset_path)
    except Exception as exc:
        print(f"Skipping dataset: {exc}")
        return False

    if sample:
        gs = gs.head(sample).reset_index(drop=True)
        print(f"Using sample of {len(gs)} pairs\n")

    # ── 1. Sensitivity analysis ────────────────────────────────────
    print("\n" + "=" * 60)
    print("1. THRESHOLD SENSITIVITY ANALYSIS  (title only)")
    print("=" * 60)
    df_results, sims = sensitivity_analysis(gs, service, thresholds)
    print("\nResults table:")
    print(df_results.to_string(index=False))

    best_row = df_results.loc[df_results["f1"].idxmax()]
    best_t   = float(best_row["threshold"])
    print(f"\n→ Best: θ={best_t:.2f}  F1={best_row['f1']:.3f}  "
          f"P={best_row['precision']:.3f}  R={best_row['recall']:.3f}")

    # ── 2. Cross-encoder ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. CROSS-ENCODER RERANKING  (title only)")
    print("=" * 60)
    ce_result = cross_encoder_evaluation(gs, service, bi_encoder_threshold=best_t)

    # ── 3. Baselines ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. BASELINE COMPARISONS")
    print("=" * 60)
    exact = exact_match_baseline(gs)
    fuzzy = fuzzy_baseline(gs, fuzzy_threshold=0.80)

    sbert = {
        "method": f"SBERT title-only (θ={best_t})",
        **{k: best_row[k] for k in ["precision", "recall", "f1", "tp", "fp", "fn", "tn"]}
    }

    comparison_rows = [exact, fuzzy, sbert]
    if ce_result:
        comparison_rows.append(ce_result)

    # ── 4. Multi-column ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. MULTI-COLUMN  (title + brand)")
    print("=" * 60)
    mc = multicolumn_evaluation(gs, service, best_t, columns=["title", "brand"])
    print(f"  P={mc['precision']}  R={mc['recall']}  F1={mc['f1']}")
    comparison_rows.append(mc)

    full_comparison = pd.DataFrame(comparison_rows)
    print("\nFull comparison (all methods):")
    print(full_comparison[["method", "precision", "recall", "f1", "tp", "fp", "fn"]].to_string(index=False))

    # ── 5. Error analysis ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("5. ERROR ANALYSIS  (best threshold)")
    print("=" * 60)
    error_analysis(gs, sims, threshold=best_t, n_samples=5)

    # ── Save ───────────────────────────────────────────────────────
    slug = _dataset_slug(dataset_path)
    df_results.to_csv(f"sensitivity_{slug}.csv", index=False)
    full_comparison.to_csv(f"comparison_{slug}.csv", index=False)
    print(f"\nSaved: sensitivity_{slug}.csv")
    print(f"Saved: comparison_{slug}.csv")

    print("\n" + "=" * 60)
    print(f"DONE: {dataset_path}")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gs", type=str, nargs="+",
        default=[
            "datasets/computers_gs.json",
            "datasets/watches_gs.json",
            "datasets/cameras_gs.json",
            "datasets/shoes_gs.json",
        ],
    )
    parser.add_argument("--sample",       type=int,  default=None)
    parser.add_argument("--model",        type=str,  default="all-mpnet-base-v2")
    parser.add_argument("--batch-size",   type=int,  default=1024)
    parser.add_argument("--cross-encoder", action="store_true",
                        help="Enable cross-encoder reranking for higher precision.")
    args = parser.parse_args()
    # print(f"DEBUG: cross_encoder flag = {args.cross_encoder}")
    # print(f"DEBUG: service.detector.cross_encoder = {service.detector.cross_encoder}")

    service = SemanticDuplicateRemoverService(
        model_name=args.model,
        threshold=0.70,
        k_neighbors=10,
        batch_size=args.batch_size,
    )
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    completed = 0
    for dataset_path in args.gs:
        if evaluate_dataset(dataset_path, service, thresholds, sample=args.sample):
            completed += 1

    print("\n" + "#" * 72)
    print(f"Completed {completed}/{len(args.gs)} dataset(s)")
    print("#" * 72)


if __name__ == "__main__":
    main()