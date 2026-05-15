"""
cluster_resolver.py
===================
Resolves ambiguous clusters using a single batched LLM call per column.

For auto-resolved clusters the mapping is built directly from
:meth:`Cluster.mapping` without any LLM involvement.

For ambiguous clusters the LLM receives all of them in ONE prompt and
returns a JSON array with its decision for each group.  This means the
number of LLM calls per column is at most 1, regardless of how many
unique values the column has.

The ValidationLayer still gates every accepted mapping before it is
written to the dataframe, so the LLM cannot bypass domain rules.
"""

from __future__ import annotations

import json
import unicodedata
from typing import Optional

try:
    from .value_clusterer import Cluster
    from .validation_layer import ValidationLayer
except ImportError:
    from value_clusterer import Cluster
    from validation_layer import ValidationLayer


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_cluster_prompt(
    column: str,
    ambiguous_clusters: list[Cluster],
    allowed_values: list[str],
) -> str:
    groups = []
    for idx, cluster in enumerate(ambiguous_clusters):
        groups.append({
            "group_id": idx,
            "members": cluster.members,
            "suggested_canonical": cluster.canonical,
        })

    allowed_section = (
        f"\nAllowed canonical values (you MUST pick from this list):\n"
        f"{json.dumps(allowed_values, ensure_ascii=False)}\n"
        if allowed_values
        else "\nNo fixed whitelist — choose the most natural canonical form.\n"
    )

    return f"""You are resolving ambiguous value groups for a single dataset column.

Column: {column}{allowed_section}
Your task: for each group decide
  1. Whether all members genuinely refer to the same concept.
  2. What the canonical form should be.
  3. Your confidence (0.0–1.0).

STRICT RULES:
- If members refer to DIFFERENT concepts, set "reject": true and explain.
- If allowed values are given, canonical MUST be one of them.
- Do NOT invent canonical values outside the allowed list.
- Do NOT merge values that have different meanings just because they look similar.
- Keep your reason short (one sentence).

Return ONLY a valid JSON array, one object per group:
[
  {{
    "group_id": <int>,
    "canonical": "<chosen canonical value>",
    "confidence": <float 0.0–1.0>,
    "reject": <true|false>,
    "reason": "<one-sentence explanation>"
  }}
]

Groups to resolve:
{json.dumps(groups, ensure_ascii=False, indent=2)}"""


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_llm_response(raw: str, n_groups: int) -> list[dict]:
    """Parse LLM JSON response, tolerating markdown fences."""
    import re
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            raise ValueError("No valid JSON array found in LLM response")

    if not isinstance(parsed, list):
        raise ValueError("LLM response is not a JSON array")

    # Index by group_id for safe lookup
    by_id: dict[int, dict] = {}
    for item in parsed:
        gid = item.get("group_id")
        if gid is not None:
            by_id[int(gid)] = item

    # Fill missing groups with a rejection placeholder
    result = []
    for i in range(n_groups):
        result.append(by_id.get(i, {
            "group_id": i,
            "canonical": None,
            "confidence": 0.0,
            "reject": True,
            "reason": "missing from LLM response",
        }))
    return result


# ---------------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------------

class ClusterResolver:
    """
    Resolves clusters produced by :func:`cluster_column` into a concrete
    value mapping ``{original_value: canonical_value}``.

    Auto-resolved clusters are mapped directly.
    Ambiguous clusters trigger a single batched LLM call.
    All mappings pass through the ValidationLayer before acceptance.
    """

    def __init__(
        self,
        chat_fn,           # callable(messages, completion_tokens) -> completion object
        validation: ValidationLayer,
        confidence_threshold: float = 0.7,
    ):
        self._chat = chat_fn
        self.validation = validation
        self.threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def resolve(
        self,
        column: str,
        auto_clusters: list[Cluster],
        ambiguous_clusters: list[Cluster],
        allowed_values: list[str],
    ) -> tuple[dict[str, str], list[dict]]:
        """
        Parameters
        ----------
        column            : column name (for validation and logging).
        auto_clusters     : clusters that can be mapped without LLM.
        ambiguous_clusters: clusters that need LLM arbitration.
        allowed_values    : whitelist (may be empty).

        Returns
        -------
        mapping : {original_value: canonical_value}
            Only contains values where the change was accepted.
        log     : list of decision records for the validation_log.
        """
        mapping: dict[str, str] = {}
        log: list[dict] = []

        # --- Auto-resolved clusters ---
        for cluster in auto_clusters:
            for original, canonical in cluster.mapping().items():
                val_ok, val_reason = self.validation.validate(column, original, canonical)
                entry = {
                    "column": column,
                    "stage": "cluster_auto",
                    "original": original,
                    "canonical": canonical,
                    "cluster_mode": cluster.mode,
                    "confidence": 0.97,
                    "accepted": val_ok,
                    "reason": cluster.reason,
                    "validation_passed": val_ok,
                    "validation_reason": val_reason,
                }
                if val_ok:
                    mapping[original] = canonical
                else:
                    entry["fallback_reason"] = f"validation failed: {val_reason}"
                log.append(entry)

        # --- Ambiguous clusters: one batched LLM call ---
        if ambiguous_clusters:
            llm_decisions = self._call_llm(column, ambiguous_clusters, allowed_values)
            for cluster, decision in zip(ambiguous_clusters, llm_decisions):
                self._apply_llm_decision(column, cluster, decision, allowed_values, mapping, log)

        return mapping, log

    # ------------------------------------------------------------------
    # LLM call (batched)
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        column: str,
        clusters: list[Cluster],
        allowed_values: list[str],
    ) -> list[dict]:
        prompt = _build_cluster_prompt(column, clusters, allowed_values)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a data-quality expert. "
                    "Return ONLY a valid JSON array, no prose."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            completion = self._chat(messages, completion_tokens=512)
            raw = completion.choices[0].message.content
            return _parse_llm_response(raw, len(clusters))
        except Exception as exc:
            # On failure, reject all ambiguous clusters in this batch
            return [
                {
                    "group_id": i,
                    "canonical": None,
                    "confidence": 0.0,
                    "reject": True,
                    "reason": f"LLM call failed: {exc}",
                }
                for i in range(len(clusters))
            ]

    # ------------------------------------------------------------------
    # Apply one LLM decision to the mapping + log
    # ------------------------------------------------------------------

    def _apply_llm_decision(
        self,
        column: str,
        cluster: Cluster,
        decision: dict,
        allowed_values: list[str],
        mapping: dict[str, str],
        log: list[dict],
    ) -> None:
        canonical_raw = decision.get("canonical") or ""
        canonical = unicodedata.normalize("NFKC", str(canonical_raw).strip())
        confidence = float(decision.get("confidence", 0.0))
        rejected = bool(decision.get("reject", False))
        llm_reason = str(decision.get("reason", ""))

        allowed_set = set(allowed_values)

        for original in cluster.members:
            if original == canonical:
                # No change needed
                log.append({
                    "column": column,
                    "stage": "cluster_llm",
                    "original": original,
                    "canonical": canonical,
                    "cluster_mode": cluster.mode,
                    "confidence": confidence,
                    "accepted": False,
                    "reason": llm_reason,
                    "fallback_reason": "original equals canonical",
                    "validation_passed": True,
                    "validation_reason": "no change",
                })
                continue

            # Rejection gate
            if rejected:
                log.append({
                    "column": column,
                    "stage": "cluster_llm",
                    "original": original,
                    "canonical": canonical or original,
                    "cluster_mode": cluster.mode,
                    "confidence": confidence,
                    "accepted": False,
                    "reason": llm_reason,
                    "fallback_reason": "LLM rejected cluster merge",
                    "validation_passed": False,
                    "validation_reason": "rejected by LLM",
                })
                continue

            # Allowed-values gate
            if allowed_set and canonical not in allowed_set:
                log.append({
                    "column": column,
                    "stage": "cluster_llm",
                    "original": original,
                    "canonical": canonical,
                    "cluster_mode": cluster.mode,
                    "confidence": confidence,
                    "accepted": False,
                    "reason": llm_reason,
                    "fallback_reason": f"LLM canonical '{canonical}' not in allowed_values",
                    "validation_passed": False,
                    "validation_reason": "canonical not in whitelist",
                })
                continue

            # Confidence gate
            if confidence < self.threshold:
                log.append({
                    "column": column,
                    "stage": "cluster_llm",
                    "original": original,
                    "canonical": canonical,
                    "cluster_mode": cluster.mode,
                    "confidence": confidence,
                    "accepted": False,
                    "reason": llm_reason,
                    "fallback_reason": (
                        f"confidence {confidence:.2f} < threshold {self.threshold:.2f}"
                    ),
                    "validation_passed": False,
                    "validation_reason": "below confidence threshold",
                })
                continue

            # Validation gate
            val_ok, val_reason = self.validation.validate(column, original, canonical)
            if val_ok:
                mapping[original] = canonical
            entry = {
                "column": column,
                "stage": "cluster_llm",
                "original": original,
                "canonical": canonical,
                "cluster_mode": cluster.mode,
                "confidence": confidence,
                "accepted": val_ok,
                "reason": llm_reason,
                "fallback_reason": "" if val_ok else f"validation failed: {val_reason}",
                "validation_passed": val_ok,
                "validation_reason": val_reason,
            }
            log.append(entry)