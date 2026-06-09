"""
Benchmark evaluation of SemanticDuplicateRemoverService
using WDC product-matching pair datasets.

Usage:
    python test_semantic_duplicates.py
    python test_semantic_duplicates.py --gs duplicates/computers_gs.json duplicates/watches_gs.json.gz duplicates/cameras_gs.json.gz duplicates/shoes_gs.json.gz
    python test_semantic_duplicates.py --sample 500
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
    return dict(precision=round(precision,3), recall=round(recall,3),
                f1=round(f1,3), tp=tp, fp=fp, fn=fn, tn=tn)


def encode_paired_texts(
    left_texts: list[str],
    right_texts: list[str],
    service: SemanticDuplicateRemoverService,
) -> tuple[np.ndarray, np.ndarray]:
    all_texts = pd.Series(left_texts + right_texts, dtype="string").fillna("").astype(str)
    unique_texts = all_texts.drop_duplicates().tolist()
    print(f"Encoding {len(unique_texts)} unique texts for {len(left_texts)} pairs ...")

    embeddings = service._encode(unique_texts)
    embedding_by_text = dict(zip(unique_texts, embeddings))
    emb_left  = np.vstack([embedding_by_text[text] for text in left_texts])
    emb_right = np.vstack([embedding_by_text[text] for text in right_texts])
    return emb_left, emb_right


def build_token_filter_mask(left_texts: list[str], right_texts: list[str]) -> np.ndarray:
    """
    Returns a boolean array of length N where True means the pair has
    conflicting discriminative tokens (product-line variant → not a duplicate).
    Apply this AFTER the similarity threshold: zero out predictions where
    conflict=True.
    """
    conflict = np.array([
        SemanticDuplicateRemoverService._discriminative_tokens_conflict(l, r)
        for l, r in zip(left_texts, right_texts)
    ])
    n_conflicts = conflict.sum()
    if n_conflicts > 0:
        print(f"  Token filter will suppress {n_conflicts} product-variant pairs.")
    return conflict


# ------------------------------------------------------------------ #
#  1. Threshold sensitivity  (SBERT, title only)                     #
# ------------------------------------------------------------------ #

def sensitivity_analysis(
    gs: pd.DataFrame,
    service: SemanticDuplicateRemoverService,
    thresholds: list[float]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Encode pairs once, sweep thresholds, return:
      - results DataFrame
      - raw similarities
      - filtered similarities (conflicts zeroed out so they never cross threshold)
    """
    emb_left, emb_right = encode_paired_texts(
        gs["title_left"].tolist(),
        gs["title_right"].tolist(),
        service,
    )

    sims   = (emb_left * emb_right).sum(axis=1)
    labels = gs["label"].values

    # Build conflict mask once for all thresholds
    conflict_mask = build_token_filter_mask(
        gs["title_left"].tolist(),
        gs["title_right"].tolist(),
    )

    # Filtered sims: set conflicting pairs to 0 so they never exceed any threshold
    sims_filtered = sims.copy()
    sims_filtered[conflict_mask] = 0.0

    rows_raw      = []
    rows_filtered = []

    for t in thresholds:
        # Raw (no filter)
        preds_raw = (sims >= t).astype(int)
        m_raw = _metrics(labels, preds_raw)
        m_raw["threshold"] = t
        rows_raw.append(m_raw)

        # Filtered
        preds_filtered = (sims_filtered >= t).astype(int)
        m_f = _metrics(labels, preds_filtered)
        m_f["threshold"] = t
        rows_filtered.append(m_f)

        print(f"  θ={t:.2f} │ "
              f"Raw:      P={m_raw['precision']:.3f}  R={m_raw['recall']:.3f}  F1={m_raw['f1']:.3f}  "
              f"(TP={m_raw['tp']}, FP={m_raw['fp']}, FN={m_raw['fn']})")
        print(f"         │ "
              f"Filtered: P={m_f['precision']:.3f}  R={m_f['recall']:.3f}  F1={m_f['f1']:.3f}  "
              f"(TP={m_f['tp']}, FP={m_f['fp']}, FN={m_f['fn']})")

    cols = ["threshold", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]
    df_raw      = pd.DataFrame(rows_raw)[cols]
    df_filtered = pd.DataFrame(rows_filtered)[cols]

    return df_raw, df_filtered, sims, sims_filtered


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
) -> tuple[dict, dict]:
    """Returns (raw_metrics, filtered_metrics)."""

    def combine(side: str) -> list[str]:
        parts = [gs[f"{c}_{side}"].fillna("").astype(str) for c in columns
                 if f"{c}_{side}" in gs.columns]
        combined = parts[0]
        for p in parts[1:]:
            combined = combined + " | " + p
        return combined.tolist()

    col_str = " + ".join(columns)
    print(f"\nMulti-column encoding ({col_str}) ...")

    left_combined  = combine("left")
    right_combined = combine("right")

    emb_left, emb_right = encode_paired_texts(left_combined, right_combined, service)
    sims   = (emb_left * emb_right).sum(axis=1)
    labels = gs["label"].values

    # Use title columns for token filter (brand names alone aren't discriminative)
    conflict_mask  = build_token_filter_mask(
        gs["title_left"].tolist(),
        gs["title_right"].tolist(),
    )
    sims_filtered  = sims.copy()
    sims_filtered[conflict_mask] = 0.0

    preds_raw      = (sims >= threshold).astype(int)
    preds_filtered = (sims_filtered >= threshold).astype(int)

    m_raw = _metrics(labels, preds_raw)
    m_f   = _metrics(labels, preds_filtered)

    method_base = f"SBERT multi-col ({col_str}) θ={threshold}"
    return (
        {"method": method_base,            **m_raw},
        {"method": method_base + "+filter", **m_f},
    )


# ------------------------------------------------------------------ #
#  4. Error analysis                                                  #
# ------------------------------------------------------------------ #

def error_analysis(
    gs: pd.DataFrame,
    sims: np.ndarray,
    threshold: float,
    n_samples: int = 5,
    label: str = ""
):
    labels = gs["label"].values
    preds  = (sims >= threshold).astype(int)

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
    df_raw, df_filtered, sims, sims_filtered = sensitivity_analysis(gs, service, thresholds)

    print("\nRaw results table:")
    print(df_raw.to_string(index=False))
    print("\nFiltered results table (token filter applied):")
    print(df_filtered.to_string(index=False))

    best_row_raw = df_raw.loc[df_raw["f1"].idxmax()]
    best_row_f   = df_filtered.loc[df_filtered["f1"].idxmax()]
    best_t_raw   = float(best_row_raw["threshold"])
    best_t_f     = float(best_row_f["threshold"])

    print(f"\n→ Best raw:      θ={best_t_raw:.2f}  F1={best_row_raw['f1']:.3f}  "
          f"P={best_row_raw['precision']:.3f}  R={best_row_raw['recall']:.3f}")
    print(f"→ Best filtered: θ={best_t_f:.2f}  F1={best_row_f['f1']:.3f}  "
          f"P={best_row_f['precision']:.3f}  R={best_row_f['recall']:.3f}")

    # Use filtered best threshold for downstream comparisons
    best_t = best_t_f

    # ── 2. Baselines ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. BASELINE COMPARISONS")
    print("=" * 60)
    exact = exact_match_baseline(gs)
    fuzzy = fuzzy_baseline(gs, fuzzy_threshold=0.80)

    sbert_raw = {
        "method": f"SBERT title-only raw (θ={best_t})",
        **{k: df_raw.loc[df_raw["threshold"] == best_t].iloc[0][k]
           for k in ["precision", "recall", "f1", "tp", "fp", "fn", "tn"]}
    }
    sbert_filtered = {
        "method": f"SBERT title-only +filter (θ={best_t})",
        **{k: df_filtered.loc[df_filtered["threshold"] == best_t].iloc[0][k]
           for k in ["precision", "recall", "f1", "tp", "fp", "fn", "tn"]}
    }

    comparison = pd.DataFrame([exact, fuzzy, sbert_raw, sbert_filtered])
    print("\n", comparison.to_string(index=False))

    # ── 3. Multi-column ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. MULTI-COLUMN  (title + brand)")
    print("=" * 60)
    mc_raw, mc_filtered = multicolumn_evaluation(gs, service, best_t, columns=["title", "brand"])
    print(f"  Raw:      P={mc_raw['precision']}  R={mc_raw['recall']}  F1={mc_raw['f1']}")
    print(f"  Filtered: P={mc_filtered['precision']}  R={mc_filtered['recall']}  F1={mc_filtered['f1']}")

    full_comparison = pd.DataFrame([exact, fuzzy, sbert_raw, sbert_filtered, mc_raw, mc_filtered])
    print("\nFull comparison (all methods):")
    print(full_comparison[["method","precision","recall","f1","tp","fp","fn"]].to_string(index=False))

    # ── 4. Error analysis ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. ERROR ANALYSIS  (best threshold, filtered)")
    print("=" * 60)
    error_analysis(gs, sims,          threshold=best_t, n_samples=5, label="raw")
    error_analysis(gs, sims_filtered, threshold=best_t, n_samples=5, label="filtered")

    # ── Save CSVs ──────────────────────────────────────────────────
    slug = _dataset_slug(dataset_path)
    df_raw.to_csv(f"sensitivity_raw_{slug}.csv",      index=False)
    df_filtered.to_csv(f"sensitivity_filtered_{slug}.csv", index=False)
    full_comparison.to_csv(f"comparison_{slug}.csv",  index=False)
    print(f"Saved: sensitivity_raw_{slug}.csv")
    print(f"Saved: sensitivity_filtered_{slug}.csv")
    print(f"Saved: comparison_{slug}.csv")

    print("\n" + "=" * 60)
    print(f"DONE: {dataset_path}")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gs",
        type=str,
        nargs="+",
        default=[
            "duplicates/computers_gs.json",
            "duplicates/watches_gs.json.gz",
            "duplicates/cameras_gs.json.gz",
            "duplicates/shoes_gs.json.gz",
        ],
        help="One or more WDC JSON/JSON.GZ pair dataset paths.",
    )
    parser.add_argument("--sample", type=int, default=None,
                        help="Use only first N pairs per dataset (quick smoke-test)")
    parser.add_argument("--model", type=str, default="all-mpnet-base-v2",
                        help="SentenceTransformer model name.")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Embedding batch size.")
    args = parser.parse_args()

    service = SemanticDuplicateRemoverService(
        threshold=0.85,
        k_neighbors=10,
        model_name=args.model,
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
