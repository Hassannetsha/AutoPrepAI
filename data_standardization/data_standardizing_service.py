import json
import re
import time
import unicodedata
from collections import deque
from difflib import SequenceMatcher
from typing import Optional

import pandas as pd
from groq import Groq

from .normalization_decision import NormalizationDecision
from .validation_layer import ValidationLayer


class DataStandardizingService:
    DEFAULT_CONFIDENCE_THRESHOLD = 0.6

    def __init__(
        self,
        df: pd.DataFrame,
        client: Groq,
        model: str,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        validation_layer: Optional[ValidationLayer] = None,
        requests_per_minute: int = 20,
        tokens_per_minute: int = 30000,
        max_retries: int = 5,
        max_completion_tokens: int = 256,
    ):
        self.df = df.copy()
        self.original_df = df.copy()
        self.client = client
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.validation = validation_layer or ValidationLayer()
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.max_retries = max_retries
        self.max_completion_tokens = max_completion_tokens

        self.results = {
            "numeric_invalid": {},
            "categorical": {},
            "llm_normalization": {},
            "validation_log": [],
        }
        self._normalization_cache: dict[tuple[str, str, float], NormalizationDecision] = {}
        self._request_times: deque[float] = deque()
        self._token_times: deque[tuple[float, int]] = deque()

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    def _parse_json_any(self, raw_text: str):
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group())

        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group())

        raise ValueError("No valid JSON found in LLM output")

    def _parse_json(self, raw_text: str):
        parsed = self._parse_json_any(raw_text)
        if not isinstance(parsed, list):
            raise ValueError("Expected JSON array")
        return parsed

    def _parse_json_object(self, raw_text: str):
        parsed = self._parse_json_any(raw_text)
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object")
        return parsed

    def _parse_with_repair(self, raw_text: str, expected_type: type):
        try:
            parsed = self._parse_json_any(raw_text)
            if not isinstance(parsed, expected_type):
                raise ValueError(f"Expected {expected_type.__name__}")
            return parsed
        except Exception:
            repair_prompt = (
                "Convert the following response into valid JSON only. "
                f"Return exactly one {expected_type.__name__} and preserve the original meaning.\n\n"
                f"{raw_text}"
            )
            repaired = self._chat_completion(
                [
                    {"role": "system", "content": "You repair malformed JSON. Return only valid JSON."},
                    {"role": "user", "content": repair_prompt},
                ],
                completion_tokens=384,
            )
            parsed = self._parse_json_any(repaired.choices[0].message.content)
            if not isinstance(parsed, expected_type):
                raise ValueError(f"Expected repaired {expected_type.__name__}")
            return parsed

    # ------------------------------------------------------------------
    # Rate limiting and retries
    # ------------------------------------------------------------------

    def _estimate_tokens(self, messages: list[dict], completion_tokens: Optional[int] = None) -> int:
        chars = sum(len(message.get("content", "")) for message in messages)
        prompt_tokens = max(1, chars // 4)
        completion_budget = completion_tokens or self.max_completion_tokens
        return prompt_tokens + completion_budget

    def _trim_rate_windows(self, now: float) -> None:
        cutoff = now - 60
        while self._request_times and self._request_times[0] <= cutoff:
            self._request_times.popleft()
        while self._token_times and self._token_times[0][0] <= cutoff:
            self._token_times.popleft()

    def _wait_for_rate_budget(self, estimated_tokens: int) -> None:
        while True:
            now = time.monotonic()
            self._trim_rate_windows(now)

            used_requests = len(self._request_times)
            used_tokens = sum(tokens for _, tokens in self._token_times)

            over_rpm = used_requests >= self.requests_per_minute
            over_tpm = used_tokens + estimated_tokens > self.tokens_per_minute

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

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "rate limit" in message or "too many requests" in message or "429" in message

    def _chat_completion(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        completion_tokens: Optional[int] = None,
    ):
        estimated_tokens = self._estimate_tokens(messages, completion_tokens)
        delay = 2.0

        for attempt in range(self.max_retries + 1):
            self._wait_for_rate_budget(estimated_tokens)
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )
            except Exception as exc:
                if not self._is_rate_limit_error(exc) or attempt == self.max_retries:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30.0)

        raise RuntimeError("Unreachable retry state")

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _simplify_text(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).lower().strip()
        return re.sub(r"[^a-z0-9]+", "", normalized)

    def canonicalize(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKC", str(value)).lower().strip()
        normalized = normalized.replace("less than or equal to", "<=")
        normalized = normalized.replace("greater than or equal to", ">=")
        normalized = normalized.replace("more than", ">")
        normalized = normalized.replace("less than", "<")
        normalized = re.sub(r"\s+", "", normalized)
        normalized = re.sub(r"[.,;:!?]+$", "", normalized)
        return normalized

    def _label_initials(self, label: str) -> str:
        parts = [part for part in re.split(r"[^A-Za-z0-9]+", label) if part]
        return "".join(part[0].lower() for part in parts)

    def _get_allowed_values(self, column: str) -> list[str]:
        rules = self.validation.get_rules(column)
        allowed_values = rules.get("allowed_values", [])
        return sorted(str(value) for value in allowed_values)

    def _variant_may_map_to_allowed(self, variant: str, allowed_values: list[str]) -> bool:
        variant_simple = self._simplify_text(variant)
        if not variant_simple:
            return False

        for label in allowed_values:
            label_simple = self._simplify_text(label)
            if not label_simple:
                continue
            if variant_simple in label_simple or label_simple in variant_simple:
                return True
            if SequenceMatcher(None, variant_simple, label_simple).ratio() >= 0.7:
                return True

        return False

    def _deterministic_allowed_match(self, original: str, allowed_values: list[str]):
        if not allowed_values:
            return None

        original_canonical = self.canonicalize(original)
        original_simple = self._simplify_text(original)

        exact_matches = [label for label in allowed_values if self.canonicalize(label) == original_canonical]
        if len(exact_matches) == 1:
            return exact_matches[0], 0.99, "exact canonical match to allowed label"

        simple_matches = [label for label in allowed_values if self._simplify_text(label) == original_simple]
        if len(simple_matches) == 1:
            return simple_matches[0], 0.97, "case/punctuation variant of allowed label"

        plural_matches = [
            label for label in allowed_values
            if self._simplify_text(label).rstrip("s") == original_simple.rstrip("s")
        ]
        if len(plural_matches) == 1:
            return plural_matches[0], 0.9, "singular/plural variant of allowed label"

        initials_matches = [
            label for label in allowed_values
            if original_simple and self._label_initials(label) == original_simple
        ]
        if len(initials_matches) == 1:
            return initials_matches[0], 0.84, "abbreviation matched allowed label initials"

        if len(original_simple) >= 4:
            prefix_matches = [
                label for label in allowed_values
                if self._simplify_text(label).startswith(original_simple)
            ]
            if len(prefix_matches) == 1:
                return prefix_matches[0], 0.82, "unique prefix match to allowed label"

        return None

    def _build_prompt(self, column: str, original: str, allowed_values: list[str]) -> str:
        if allowed_values:
            return f"""You are normalizing ONE categorical value for a tabular dataset.

Column name: {column if column else "unknown"}
Input value: {original}

Allowed labels:
{json.dumps(allowed_values, ensure_ascii=False)}

Rules:
- Map the input to exactly ONE allowed label when it clearly means the same thing.
- Return the original input unchanged if you are not confident enough to map it safely.
- Prefer canonical label mapping over formatting-only edits when the input is an abbreviation, typo, or synonym.
- Do NOT invent labels outside the allowed list.
- Rate your raw confidence from 0.0 to 1.0.

Return ONLY valid JSON:
{{
  "normalized_value": "string",
  "confidence": float,
  "reason": "short explanation"
}}"""

        return f"""You are normalizing ONE categorical value for a tabular dataset.

Column name: {column if column else "unknown"}
Input value: {original}

Rules:
- Keep the SAME meaning.
- Fix formatting only (case, spacing, punctuation, obvious encoding issues).
- Do NOT translate.
- Do NOT invent new information.
- Rate your raw confidence from 0.0 to 1.0.

Return ONLY valid JSON:
{{
  "normalized_value": "string",
  "confidence": float,
  "reason": "short explanation"
}}"""

    def _calibrate_confidence(
        self,
        original: str,
        normalized: str,
        raw_confidence: float,
        allowed_values: list[str],
    ) -> float:
        raw = min(max(raw_confidence, 0.0), 1.0)

        if normalized == original:
            if allowed_values and normalized in allowed_values:
                return 0.99
            return round(min(raw, 0.6), 3)

        original_canonical = self.canonicalize(original)
        normalized_canonical = self.canonicalize(normalized)
        original_simple = self._simplify_text(original)
        normalized_simple = self._simplify_text(normalized)

        if original_canonical == normalized_canonical or original_simple == normalized_simple:
            return 0.98

        similarity = SequenceMatcher(None, original_simple, normalized_simple).ratio()

        if allowed_values and normalized in allowed_values:
            if self._label_initials(normalized) == original_simple:
                return 0.84
            calibrated = 0.6 + (0.15 * similarity) + (0.25 * raw)
            return min(round(calibrated, 3), 0.95)

        calibrated = 0.45 + (0.25 * similarity) + (0.2 * raw)
        return min(round(calibrated, 3), 0.9)

    def _decision_to_log_entry(self, column: str, decision: NormalizationDecision) -> dict:
        entry = decision.to_dict().copy()
        entry["column"] = column
        entry["stage"] = "llm_normalization"
        return entry

    # ------------------------------------------------------------------
    # Core: single-value normalization
    # ------------------------------------------------------------------

    def normalize_value(
        self,
        value,
        column: str = "",
        confidence_threshold: Optional[float] = None,
    ) -> NormalizationDecision:
        threshold = self.confidence_threshold if confidence_threshold is None else confidence_threshold

        if not isinstance(value, str):
            return NormalizationDecision(
                original=str(value),
                normalized=str(value),
                confidence=1.0,
                raw_confidence=1.0,
                accepted=True,
                fallback_reason="",
                validation_passed=True,
                validation_reason="non-string passthrough",
                reason="non-string passthrough",
            )

        original = value.strip()
        if not original:
            return NormalizationDecision(
                original=original,
                normalized=original,
                confidence=1.0,
                raw_confidence=1.0,
                accepted=True,
                fallback_reason="",
                validation_passed=True,
                validation_reason="empty string passthrough",
                reason="empty string passthrough",
            )

        cache_key = (column, original, threshold)
        if cache_key in self._normalization_cache:
            return self._normalization_cache[cache_key]

        allowed_values = self._get_allowed_values(column)
        deterministic_match = self._deterministic_allowed_match(original, allowed_values)

        if deterministic_match is not None:
            normalized_value, confidence, reason = deterministic_match
            val_ok, val_reason = self.validation.validate(column, original, normalized_value)
            accepted = val_ok and confidence >= threshold
            decision = NormalizationDecision(
                original=original,
                normalized=normalized_value if accepted else original,
                confidence=confidence,
                raw_confidence=confidence,
                accepted=accepted,
                fallback_reason="" if accepted else f"validation failed: {val_reason}",
                validation_passed=val_ok,
                validation_reason=val_reason,
                reason=reason,
            )
            self._normalization_cache[cache_key] = decision
            return decision

        messages = [
            {"role": "system", "content": "You are a data quality expert. Return ONLY valid JSON."},
            {"role": "user", "content": self._build_prompt(column, original, allowed_values)},
        ]

        try:
            completion = self._chat_completion(messages)
            raw = completion.choices[0].message.content
            parsed = self._parse_with_repair(raw, dict)

            llm_normalized = str(parsed.get("normalized_value", original)).strip()
            llm_normalized = unicodedata.normalize("NFKC", llm_normalized)
            raw_confidence = float(parsed.get("confidence", 0.0))
            llm_reason = str(parsed.get("reason", "")).strip()

            if not llm_normalized:
                raise ValueError("LLM returned empty normalized_value")

        except Exception as exc:
            decision = NormalizationDecision(
                original=original,
                normalized=original,
                confidence=0.0,
                raw_confidence=0.0,
                accepted=False,
                fallback_reason=f"LLM call/parse failed: {exc}",
                validation_passed=False,
                validation_reason="validation skipped because LLM failed",
                reason="llm_error",
            )
            self._normalization_cache[cache_key] = decision
            return decision

        calibrated_confidence = self._calibrate_confidence(
            original,
            llm_normalized,
            raw_confidence,
            allowed_values,
        )
        val_ok, val_reason = self.validation.validate(column, original, llm_normalized)
        confidence_ok = calibrated_confidence >= threshold or llm_normalized == original

        if val_ok and confidence_ok:
            accepted = True
            final_value = llm_normalized
            fallback_reason = ""
        else:
            accepted = False
            final_value = original
            reasons = []
            if not confidence_ok:
                reasons.append(
                    f"confidence {calibrated_confidence:.2f} < threshold {threshold:.2f}"
                )
            if not val_ok:
                reasons.append(f"validation failed: {val_reason}")
            fallback_reason = "; ".join(reasons)

        decision = NormalizationDecision(
            original=original,
            normalized=final_value,
            confidence=calibrated_confidence,
            raw_confidence=raw_confidence,
            accepted=accepted,
            fallback_reason=fallback_reason,
            validation_passed=val_ok,
            validation_reason=val_reason,
            reason=llm_reason,
        )
        self._normalization_cache[cache_key] = decision
        return decision

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_categorical_issues(self, column: str, sample_size: int = 100):
        values = self.df[column].dropna().astype(str).unique().tolist()[:sample_size]
        prompt = f"""You are performing CATEGORICAL VALUE CLEANING.

Column: {column}

Tasks:
1. Identify values that are INVALID (do not belong to this column).
2. Identify values that need STANDARDIZATION (same meaning, different surface form).

STRICT RULES:
- Do NOT merge distinct categories.
- Do NOT create higher-level abstractions.
- Do NOT group values unless they clearly refer to the SAME label.
- Do NOT include values that are already clean.
- Add a confidence score (0.0-1.0) for each issue.

Return ONLY valid JSON:
[
  {{
    "type": "standardize" | "invalid",
    "canonical_value": "string | null",
    "variants": ["v1", "v2"],
    "confidence": float,
    "reason": "short explanation"
  }}
]

Values:
{values}"""

        try:
            completion = self._chat_completion(
                [
                    {"role": "system", "content": "You are a data quality expert. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                completion_tokens=384,
            )
            raw = completion.choices[0].message.content
            issues = self._parse_with_repair(raw, list)
        except Exception as exc:
            print(f"  [WARNING] detect_categorical_issues failed for '{column}': {exc}")
            issues = []

        for issue in issues:
            issue.setdefault("confidence", 0.5)

        self.results["categorical"][column] = issues
        return issues

    def detect_numeric_issues(self, column: str, sample_size: int = 200):
        rules = self.validation.get_rules(column)
        issues = []
        series = self.df[column].dropna().head(sample_size)

        if "min_value" in rules:
            bad_values = series[series < rules["min_value"]]
            for index, value in bad_values.items():
                issues.append(
                    {
                        "index": int(index),
                        "value": value,
                        "confidence": 1.0,
                        "reason": f"value below min {rules['min_value']}",
                    }
                )

        if "max_value" in rules:
            bad_values = series[series > rules["max_value"]]
            for index, value in bad_values.items():
                issues.append(
                    {
                        "index": int(index),
                        "value": value,
                        "confidence": 1.0,
                        "reason": f"value above max {rules['max_value']}",
                    }
                )

        if issues:
            deduped = {}
            for issue in issues:
                deduped[(issue["index"], issue["value"])] = issue
            issues = list(deduped.values())
            self.results["numeric_invalid"][column] = issues
            return issues

        values = series.tolist()
        prompt = f"""You are analyzing a NUMERIC column for INVALID values only.

Column: {column}

Rules:
- ONLY flag values that are logically impossible or nonsensical.
- DO NOT flag outliers, rare values, or extreme but possible values.
- Add a confidence score (0.0-1.0) per flagged value.

Return ONLY valid JSON:
[
  {{
    "index": int,
    "value": number,
    "confidence": float,
    "reason": "why this value is invalid"
  }}
]

Values (index, value):
{list(enumerate(values))}"""

        try:
            completion = self._chat_completion(
                [
                    {"role": "system", "content": "You are a data quality expert. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                completion_tokens=384,
            )
            raw = completion.choices[0].message.content
            issues = self._parse_with_repair(raw, list)
        except Exception as exc:
            print(f"  [WARNING] detect_numeric_issues failed for '{column}': {exc}")
            issues = []

        for issue in issues:
            issue.setdefault("confidence", 0.5)

        self.results["numeric_invalid"][column] = issues
        return issues

    def run_detection(self, columns: Optional[list[str]] = None):
        target_columns = columns or list(self.df.columns)
        for col in target_columns:
            print(f"  Analyzing column: {col}")
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.detect_numeric_issues(col)
            else:
                self.detect_categorical_issues(col)

    # ------------------------------------------------------------------
    # Apply fixes
    # ------------------------------------------------------------------

    def apply_categorical_fixes(
        self,
        columns=None,
        apply_invalid: bool = False,
        confidence_threshold: Optional[float] = None,
    ):
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        columns = columns or list(self.results["categorical"].keys())

        for col in columns:
            allowed_values = self._get_allowed_values(col)
            for issue in self.results["categorical"].get(col, []):
                conf = issue.get("confidence", 0.0)
                if conf < threshold:
                    self.results["validation_log"].append(
                        {
                            "stage": "apply_categorical_fixes",
                            "column": col,
                            "action": "skipped",
                            "issue": issue,
                            "reason": f"confidence {conf:.2f} < threshold {threshold:.2f}",
                        }
                    )
                    continue

                if issue["type"] == "standardize" and issue.get("canonical_value"):
                    canonical = str(issue["canonical_value"]).strip()
                    valid_variants = []
                    skipped_variants = []
                    for variant in issue["variants"]:
                        val_ok, val_reason = self.validation.validate(col, variant, canonical)
                        if val_ok:
                            valid_variants.append(variant)
                        else:
                            skipped_variants.append({"variant": variant, "reason": val_reason})

                    if not valid_variants:
                        self.results["validation_log"].append(
                            {
                                "stage": "apply_categorical_fixes",
                                "column": col,
                                "action": "skipped",
                                "issue": issue,
                                "reason": "canonical value failed validation for all variants",
                            }
                        )
                        continue

                    for variant in valid_variants:
                        self.df[col] = self.df[col].replace(variant, canonical)
                    self.results["validation_log"].append(
                        {
                            "stage": "apply_categorical_fixes",
                            "column": col,
                            "action": "standardized",
                            "variants": valid_variants,
                            "canonical": canonical,
                            "confidence": conf,
                            "reason": issue.get("reason", ""),
                            "skipped_variants": skipped_variants,
                        }
                    )
                elif apply_invalid and issue["type"] == "invalid":
                    safe_to_nullify = []
                    deferred_variants = []
                    for variant in issue["variants"]:
                        if allowed_values and self._variant_may_map_to_allowed(variant, allowed_values):
                            deferred_variants.append(variant)
                            continue
                        safe_to_nullify.append(variant)

                    for variant in safe_to_nullify:
                        self.df[col] = self.df[col].replace(variant, pd.NA)
                    self.results["validation_log"].append(
                        {
                            "stage": "apply_categorical_fixes",
                            "column": col,
                            "action": "nullified_invalid" if safe_to_nullify else "skipped",
                            "variants": safe_to_nullify,
                            "confidence": conf,
                            "reason": issue.get("reason", ""),
                            "deferred_variants": deferred_variants,
                        }
                    )

    def apply_llm_normalization(
        self,
        columns=None,
        max_unique_values: int = 200,
        confidence_threshold: Optional[float] = None,
    ):
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold

        if columns is None:
            columns = [
                col for col in self.df.columns
                if not pd.api.types.is_numeric_dtype(self.df[col])
            ]

        for col in columns:
            print(f"  Normalizing column: {col}")
            values = self.df[col].dropna().astype(str)
            unique_values = values.unique().tolist()[:max_unique_values]

            mapping = {}
            col_log = []

            for raw_value in unique_values:
                decision = self.normalize_value(
                    raw_value,
                    column=col,
                    confidence_threshold=threshold,
                )
                col_log.append(self._decision_to_log_entry(col, decision))

                if decision.accepted and decision.normalized != raw_value:
                    mapping[raw_value] = decision.normalized

            if mapping:
                self.df[col] = self.df[col].replace(mapping)

            self.results["llm_normalization"][col] = {
                "accepted_changes": mapping,
                "total_unique_checked": len(unique_values),
                "total_accepted": len(mapping),
                "total_rejected": len(unique_values) - len(mapping),
            }
            self.results["validation_log"].extend(col_log)

        return self.results["llm_normalization"]

    # ------------------------------------------------------------------
    # Summary and evaluation
    # ------------------------------------------------------------------

    def summary(self):
        print("=" * 60)
        print("FINAL SUMMARY OF DETECTED ISSUES")
        print("=" * 60)

        print("\n### Numeric Invalid Values ###")
        for col, issues in self.results["numeric_invalid"].items():
            above = [i for i in issues if i.get("confidence", 0) >= self.confidence_threshold]
            print(f"  {col}: {len(issues)} flagged, {len(above)} above confidence threshold")

        print("\n### Categorical Standardization Issues ###")
        for col, issues in self.results["categorical"].items():
            above = [i for i in issues if i.get("confidence", 0) >= self.confidence_threshold]
            print(f"  {col}: {len(issues)} flagged, {len(above)} above confidence threshold")

        print("\n### LLM Normalization Summary ###")
        for col, info in self.results["llm_normalization"].items():
            print(
                f"  {col}: checked={info['total_unique_checked']}, "
                f"accepted={info['total_accepted']}, "
                f"rejected={info['total_rejected']}"
            )

        applied_fix_count = sum(
            1
            for entry in self.results["validation_log"]
            if entry.get("action") in {"standardized", "nullified_invalid"}
        )
        print(f"\n### Applied Fix Actions ###\n  total_actions={applied_fix_count}")

        print("\n### Validation Log (last 10 entries) ###")
        for entry in self.results["validation_log"][-10:]:
            print(f"  {entry}")

        print("=" * 60)
        return self.results

    def _resolve_mapping(self, value: str, mapping: dict[str, str], max_hops: int = 5) -> str:
        current = value
        seen = set()
        for _ in range(max_hops):
            nxt = mapping.get(current)
            if not nxt or nxt in seen:
                break
            seen.add(current)
            current = nxt
        return current

    def _build_final_mapping(self, column: str) -> dict[str, str]:
        final_mapping = {}

        for entry in self.results["validation_log"]:
            if entry.get("column") != column:
                continue
            if entry.get("stage") == "apply_categorical_fixes" and entry.get("action") == "standardized":
                for variant in entry.get("variants", []):
                    final_mapping[variant] = entry["canonical"]

        final_mapping.update(
            self.results["llm_normalization"].get(column, {}).get("accepted_changes", {})
        )
        return {source: self._resolve_mapping(target, final_mapping) for source, target in final_mapping.items()}

    def _observed_original_values(self, column: str) -> set[str]:
        if column not in self.original_df.columns:
            return set()
        values = self.original_df[column].dropna().astype(str)
        return {value.strip() for value in values.tolist()}

    def evaluate(self, ground_truth: dict[str, dict[str, str]]) -> dict:
        overall = {"tp": 0, "fp": 0, "fn": 0, "fallbacks": 0, "total": 0}
        per_column = {}

        for col, gt_map in ground_truth.items():
            final_mapping = self._build_final_mapping(col)
            observed_values = self._observed_original_values(col)
            filtered_gt_map = {
                original: expected
                for original, expected in gt_map.items()
                if not observed_values or original in observed_values
            }

            tp = fp = fn = fallbacks = 0
            calibration = []

            for original, expected in filtered_gt_map.items():
                overall["total"] += 1

                decision_data = next(
                    (
                        entry for entry in self.results["validation_log"]
                        if entry.get("stage") == "llm_normalization"
                        and entry.get("column") == col
                        and entry.get("original") == original
                    ),
                    None,
                )

                predicted = final_mapping.get(original, original)
                is_correct = self.canonicalize(predicted) == self.canonicalize(expected)
                changed = original in final_mapping

                if changed and is_correct:
                    tp += 1
                elif changed and not is_correct:
                    fp += 1
                elif not changed and self.canonicalize(expected) != self.canonicalize(original):
                    fn += 1
                    fallbacks += 1

                if decision_data:
                    calibration.append((decision_data["confidence"], is_correct))

            precision = tp / (tp + fp) if (tp + fp) > 0 else None
            recall = tp / (tp + fn) if (tp + fn) > 0 else None
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision is not None and recall is not None and (precision + recall) > 0
                else 0.0
            )

            conf_correct = [c for c, ok in calibration if ok]
            conf_incorrect = [c for c, ok in calibration if not ok]

            per_column[col] = {
                "precision": round(precision, 3) if precision is not None else "N/A",
                "recall": round(recall, 3) if recall is not None else "N/A",
                "f1": round(f1, 3),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "fallback_rate": round(fallbacks / len(filtered_gt_map), 3) if filtered_gt_map else 0,
                "mean_conf_correct": round(sum(conf_correct) / len(conf_correct), 3) if conf_correct else "N/A",
                "mean_conf_incorrect": round(sum(conf_incorrect) / len(conf_incorrect), 3) if conf_incorrect else "N/A",
                "total_evaluated": len(filtered_gt_map),
            }

            overall["tp"] += tp
            overall["fp"] += fp
            overall["fn"] += fn
            overall["fallbacks"] += fallbacks

        tp, fp, fn = overall["tp"], overall["fp"], overall["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0
            else 0.0
        )

        results = {
            "per_column": per_column,
            "overall": {
                "precision": round(precision, 3) if precision is not None else "N/A",
                "recall": round(recall, 3) if recall is not None else "N/A",
                "f1": round(f1, 3),
                "fallback_rate": round(overall["fallbacks"] / overall["total"], 3) if overall["total"] else 0,
                "total_evaluated": overall["total"],
            },
        }

        print("\n=== EVALUATION RESULTS ===")
        print(json.dumps(results, indent=2))
        return results
