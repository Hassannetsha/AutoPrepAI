"""
data_standardizing_service.py
==============================
Unified data standardization service.

Pipeline (per column)
---------------------
Numeric columns
    Rule-based validation (min/max) → flag invalids → NaN-replace or report.
    If no rules: LLM detects impossible values.

Categorical columns
    1. cluster_column()   — safe edit-distance graph groups unique values.
    2. split_clusters()   — separates auto-resolved from ambiguous clusters.
    3. ClusterResolver    — applies auto mappings directly; sends all
                           ambiguous clusters in ONE batched LLM call.
    4. ValidationLayer    — gates every mapping before it touches the df.

Entry points
------------
    svc.standardize()                 # full pipeline, recommended
    svc.standardize_column(col)       # single column
    svc.detect_numeric_issues(col)    # inspect only, no changes
    svc.summary()                     # print results
    svc.evaluate(ground_truth)        # precision / recall / F1
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections import deque
from typing import Optional

import pandas as pd
from groq import Groq

try:
    from .cluster_resolver import ClusterResolver
    from .normalization_decision import NormalizationDecision
    from .validation_layer import ValidationLayer
    from .value_clusterer import cluster_column, split_clusters
except ImportError:
    from cluster_resolver import ClusterResolver
    from normalization_decision import NormalizationDecision
    from validation_layer import ValidationLayer
    from value_clusterer import cluster_column, split_clusters


class DataStandardizingService:

    DEFAULT_CONFIDENCE_THRESHOLD = 0.7
    DEFAULT_SIMILARITY_THRESHOLD = 0.35

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        df: pd.DataFrame,
        client: Groq,
        model: str,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        validation_layer: Optional[ValidationLayer] = None,
        requests_per_minute: int = 20,
        tokens_per_minute: int = 30_000,
        max_retries: int = 5,
        max_unique_values: int = 500,
    ):
        self.df = df.copy().reset_index(drop=True)
        self.original_df = df.copy().reset_index(drop=True)
        self.client = client
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.validation = validation_layer or ValidationLayer()
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.max_retries = max_retries
        self.max_unique_values = max_unique_values

        self.results: dict = {
            "numeric_issues": {},       # col -> list of flagged rows
            "standardization": {},      # col -> {mapping, clusters, stats}
            "validation_log": [],       # every accept/reject decision
        }

        # Rate-limiting state
        self._request_times: deque[float] = deque()
        self._token_times: deque[tuple[float, int]] = deque()

    # ------------------------------------------------------------------
    # Public pipeline entry points
    # ------------------------------------------------------------------

    def standardize(
        self,
        columns: Optional[list[str]] = None,
        numeric_columns: Optional[list[str]] = None,
        categorical_columns: Optional[list[str]] = None,
    ) -> dict:
        """
        Run the full standardization pipeline.

        Parameters
        ----------
        columns            : columns to process (default: all).
                             Use this when you want automatic type detection.
        numeric_columns    : explicit list of numeric columns to validate.
        categorical_columns: explicit list of categorical columns to cluster.

        If *columns* is given, numeric/categorical split is inferred from
        pandas dtype. If *numeric_columns* / *categorical_columns* are given
        they take precedence.

        Returns self.results.
        """
        target = columns or list(self.df.columns)

        num_cols = (
            numeric_columns
            if numeric_columns is not None
            else [c for c in target if pd.api.types.is_numeric_dtype(self.df[c])]
        )
        cat_cols = (
            categorical_columns
            if categorical_columns is not None
            else [c for c in target if not pd.api.types.is_numeric_dtype(self.df[c])]
        )

        print(f"Standardizing {len(num_cols)} numeric + {len(cat_cols)} categorical columns.")

        for col in num_cols:
            print(f"\n[numeric] {col}")
            self._handle_numeric(col)

        for col in cat_cols:
            print(f"\n[categorical] {col}")
            self.standardize_column(col)

        return self.results

    def standardize_column(self, column: str) -> dict:
        """
        Run the cluster pipeline on a single categorical column.

        Returns the per-column result dict.
        """
        allowed_values = self._get_allowed_values(column)
        unique_count = self.df[column].nunique(dropna=True)

        # Guard: low-cardinality columns with no whitelist are risky to
        # auto-standardize — we don't know what the valid values are.
        if unique_count < 20 and not allowed_values:
            reason = (
                f"skipped: {unique_count} unique values, no allowed_values "
                "registered — cannot safely determine canonical forms"
            )
            print(f"  {reason}")
            return self._record_skip(column, reason)

        # Collect unique values ranked by frequency
        series = self.df[column].dropna().astype(str)
        vc = series.value_counts()
        unique_values = vc.index.tolist()[: self.max_unique_values]
        value_counts = vc.to_dict()

        # Step 1: cluster
        clusters = cluster_column(
            unique_values=unique_values,
            value_counts=value_counts,
            allowed_values=allowed_values or None,
            similarity_threshold=self.similarity_threshold,
        )
        auto_clusters, ambiguous_clusters = split_clusters(clusters)

        n_variants_auto = sum(len(c.variants) for c in auto_clusters)
        print(
            f"  clusters={len(clusters)}  "
            f"auto={len(auto_clusters)} ({n_variants_auto} variants)  "
            f"ambiguous={len(ambiguous_clusters)}"
        )

        # Step 2: resolve (auto → direct map; ambiguous → 1 LLM call)
        resolver = ClusterResolver(
            chat_fn=self._chat_completion,
            validation=self.validation,
            confidence_threshold=self.confidence_threshold,
        )
        mapping, col_log = resolver.resolve(
            column=column,
            auto_clusters=auto_clusters,
            ambiguous_clusters=ambiguous_clusters,
            allowed_values=allowed_values,
        )

        # Step 3: apply
        if mapping:
            self.df[column] = self.df[column].replace(mapping)

        # Step 4: record
        result = {
            "accepted_changes": mapping,
            "total_unique_checked": len(unique_values),
            "total_accepted": len(mapping),
            "total_rejected": len(unique_values) - len(mapping)
                              - sum(1 for c in clusters
                                    if len(c.members) == 1 and not c.ambiguous),
            "n_clusters": len(clusters),
            "n_auto_clusters": len(auto_clusters),
            "n_ambiguous_clusters": len(ambiguous_clusters),
            "llm_calls": 1 if ambiguous_clusters else 0,
            "clusters_summary": [
                {
                    "canonical": c.canonical,
                    "members": c.members,
                    "mode": c.mode,
                    "ambiguous": c.ambiguous,
                    "reason": c.reason,
                }
                for c in clusters
            ],
        }

        self.results["standardization"][column] = result
        # Keep llm_normalization key for evaluate() back-compat
        self.results.setdefault("llm_normalization", {})[column] = {
            "accepted_changes": mapping,
            "total_unique_checked": len(unique_values),
            "total_accepted": len(mapping),
            "total_rejected": len(unique_values) - len(mapping),
        }
        self.results["validation_log"].extend(col_log)

        print(
            f"  accepted={len(mapping)}  "
            f"llm_calls={result['llm_calls']}"
        )
        return result

    # ------------------------------------------------------------------
    # Numeric handling
    # ------------------------------------------------------------------

    def detect_numeric_issues(self, column: str, sample_size: int = 200) -> list[dict]:
        """
        Flag invalid numeric values.

        If the ValidationLayer has min/max rules for this column they are
        applied deterministically (no LLM).  Otherwise the LLM is asked to
        identify logically impossible values.

        Returns a list of issue dicts.
        """
        rules = self.validation.get_rules(column)
        issues: list[dict] = []
        series = self.df[column].dropna().head(sample_size)

        # --- Rule-based (fast, deterministic) ---
        if "min_value" in rules:
            for idx, val in series[series < rules["min_value"]].items():
                issues.append({
                    "index": int(idx), "value": val, "confidence": 1.0,
                    "reason": f"below min {rules['min_value']}",
                })
        if "max_value" in rules:
            for idx, val in series[series > rules["max_value"]].items():
                issues.append({
                    "index": int(idx), "value": val, "confidence": 1.0,
                    "reason": f"above max {rules['max_value']}",
                })

        if issues:
            # Deduplicate by (index, value)
            seen: dict = {}
            for iss in issues:
                seen[(iss["index"], iss["value"])] = iss
            issues = list(seen.values())
            self.results["numeric_issues"][column] = issues
            return issues

        # --- LLM fallback (no rules registered) ---
        values = list(enumerate(series.tolist()))
        prompt = (
            f"You are analyzing a NUMERIC column for INVALID values only.\n\n"
            f"Column: {column}\n\n"
            "Rules:\n"
            "- ONLY flag values that are logically impossible or nonsensical.\n"
            "- DO NOT flag outliers or rare but possible values.\n"
            "- Add confidence (0.0–1.0) per flagged value.\n\n"
            "Return ONLY valid JSON:\n"
            "[\n"
            '  {"index": int, "value": number, "confidence": float, '
            '"reason": "why invalid"}\n'
            "]\n\n"
            f"Values (index, value):\n{values}"
        )
        try:
            completion = self._chat_completion(
                [
                    {"role": "system", "content": "You are a data quality expert. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                completion_tokens=384,
            )
            issues = self._parse_json(completion.choices[0].message.content, list)
        except Exception as exc:
            print(f"  [WARNING] detect_numeric_issues LLM failed for '{column}': {exc}")
            issues = []

        for iss in issues:
            iss.setdefault("confidence", 0.5)

        self.results["numeric_issues"][column] = issues
        return issues

    def _handle_numeric(self, column: str) -> None:
        """Detect and NaN-replace confirmed invalid numeric values."""
        issues = self.detect_numeric_issues(column)
        threshold = self.confidence_threshold
        applied = 0
        for iss in issues:
            if iss.get("confidence", 0) >= threshold:
                idx = iss["index"]
                if idx in self.df.index:
                    self.df.at[idx, column] = pd.NA
                    applied += 1
                    self.results["validation_log"].append({
                        "stage": "numeric_fix",
                        "column": column,
                        "index": idx,
                        "original_value": iss["value"],
                        "action": "set_to_NA",
                        "confidence": iss["confidence"],
                        "reason": iss["reason"],
                    })
        print(f"  flagged={len(issues)}  nullified={applied}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_allowed_values(self, column: str) -> list[str]:
        rules = self.validation.get_rules(column)
        return sorted(str(v) for v in rules.get("allowed_values", []))

    def _record_skip(self, column: str, reason: str) -> dict:
        result = {"skipped": True, "skip_reason": reason}
        self.results["standardization"][column] = result
        self.results["validation_log"].append({
            "stage": "standardize_column",
            "column": column,
            "action": "skipped",
            "reason": reason,
        })
        return result

    # ------------------------------------------------------------------
    # LLM plumbing (rate limiting + retries)
    # ------------------------------------------------------------------

    def _estimate_tokens(self, messages: list[dict], completion_tokens: int) -> int:
        chars = sum(len(m.get("content", "")) for m in messages)
        return max(1, chars // 4) + completion_tokens

    def _trim_windows(self, now: float) -> None:
        cutoff = now - 60
        while self._request_times and self._request_times[0] <= cutoff:
            self._request_times.popleft()
        while self._token_times and self._token_times[0][0] <= cutoff:
            self._token_times.popleft()

    def _wait_for_budget(self, estimated_tokens: int) -> None:
        while True:
            now = time.monotonic()
            self._trim_windows(now)
            over_rpm = len(self._request_times) >= self.requests_per_minute
            used_tok = sum(t for _, t in self._token_times)
            over_tpm = used_tok + estimated_tokens > self.tokens_per_minute
            if not over_rpm and not over_tpm:
                self._request_times.append(now)
                self._token_times.append((now, estimated_tokens))
                return
            waits = []
            if over_rpm and self._request_times:
                waits.append(60 - (now - self._request_times[0]))
            if over_tpm and self._token_times:
                waits.append(60 - (now - self._token_times[0][0]))
            time.sleep(max(0.5, min(waits) if waits else 1.0))

    def _chat_completion(
        self,
        messages: list[dict],
        completion_tokens: int = 512,
        temperature: float = 0.0,
    ):
        estimated = self._estimate_tokens(messages, completion_tokens)
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            self._wait_for_budget(estimated)
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )
            except Exception as exc:
                msg = str(exc).lower()
                is_rate = "rate limit" in msg or "429" in msg or "too many requests" in msg
                if not is_rate or attempt == self.max_retries:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        raise RuntimeError("Unreachable")

    def _parse_json(self, raw: str, expected_type: type):
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, expected_type):
                raise ValueError(f"Expected {expected_type.__name__}")
            return parsed
        except Exception:
            # Try extracting the first matching bracket block
            pattern = r"\[.*\]" if expected_type is list else r"\{.*\}"
            match = re.search(pattern, raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"No valid {expected_type.__name__} JSON found")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        print("=" * 60)
        print("STANDARDIZATION SUMMARY")
        print("=" * 60)

        print("\n### Numeric Issues ###")
        for col, issues in self.results["numeric_issues"].items():
            above = [i for i in issues if i.get("confidence", 0) >= self.confidence_threshold]
            print(f"  {col}: {len(issues)} flagged, {len(above)} above threshold")

        print("\n### Categorical Standardization ###")
        for col, info in self.results["standardization"].items():
            if info.get("skipped"):
                print(f"  {col}: SKIPPED — {info['skip_reason']}")
                continue
            print(
                f"  {col}: checked={info['total_unique_checked']}  "
                f"accepted={info['total_accepted']}  "
                f"clusters={info['n_clusters']}  "
                f"llm_calls={info['llm_calls']}"
            )
            for c in info.get("clusters_summary", []):
                tag = "AUTO" if not c["ambiguous"] else "LLM "
                print(f"    [{tag}] {c['canonical']!r:25} ← {c['members']}")

        total_llm = sum(
            info.get("llm_calls", 0)
            for info in self.results["standardization"].values()
            if not info.get("skipped")
        )
        print(f"\n  Total LLM calls: {total_llm}")

        applied = sum(
            1 for e in self.results["validation_log"]
            if e.get("accepted") and e.get("stage") in ("cluster_auto", "cluster_llm")
        )
        print(f"  Total mappings applied: {applied}")
        print("=" * 60)
        return self.results

    # ------------------------------------------------------------------
    # Evaluation (unchanged interface, works with new results structure)
    # ------------------------------------------------------------------

    def evaluate(self, ground_truth: dict[str, dict[str, str]]) -> dict:
        """
        Compare applied mappings against a ground-truth dict.

        ground_truth format:
            { "column_name": { "dirty_value": "expected_canonical", ... } }

        Returns precision, recall, F1 per column and overall.
        """
        def _canon(v: str) -> str:
            s = unicodedata.normalize("NFKC", v).lower().strip()
            return re.sub(r"\s+", "", s)

        overall = {"tp": 0, "fp": 0, "fn": 0, "total": 0}
        per_column: dict = {}

        for col, gt_map in ground_truth.items():
            col_info = self.results["standardization"].get(col, {})
            applied_mapping = col_info.get("accepted_changes", {})

            # Only evaluate values actually seen in the original df
            observed = set(
                self.original_df[col].dropna().astype(str).str.strip().unique()
                if col in self.original_df.columns else []
            )
            filtered = {
                orig: expected
                for orig, expected in gt_map.items()
                if not observed or orig in observed
            }

            tp = fp = fn = 0
            for original, expected in filtered.items():
                overall["total"] += 1
                predicted = applied_mapping.get(original, original)
                correct = _canon(predicted) == _canon(expected)
                changed = original in applied_mapping

                if changed and correct:
                    tp += 1
                elif changed and not correct:
                    fp += 1
                elif not changed and _canon(expected) != _canon(original):
                    fn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else None
            recall = tp / (tp + fn) if (tp + fn) > 0 else None
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision is not None and recall is not None
                and (precision + recall) > 0
                else 0.0
            )

            per_column[col] = {
                "precision": round(precision, 3) if precision is not None else "N/A",
                "recall": round(recall, 3) if recall is not None else "N/A",
                "f1": round(f1, 3),
                "tp": tp, "fp": fp, "fn": fn,
                "total_evaluated": len(filtered),
                "fallback_rate": round(fn / len(filtered), 3) if filtered else 0,
            }
            overall["tp"] += tp
            overall["fp"] += fp
            overall["fn"] += fn

        tp, fp, fn = overall["tp"], overall["fp"], overall["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None
            and (precision + recall) > 0
            else 0.0
        )

        results = {
            "per_column": per_column,
            "overall": {
                "precision": round(precision, 3) if precision is not None else "N/A",
                "recall": round(recall, 3) if recall is not None else "N/A",
                "f1": round(f1, 3),
                "fallback_rate": round(overall["fn"] / overall["total"], 3)
                                 if overall["total"] else 0,
                "total_evaluated": overall["total"],
            },
        }

        print("\n=== EVALUATION RESULTS ===")
        print(json.dumps(results, indent=2))
        return results
