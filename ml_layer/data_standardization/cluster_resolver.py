from __future__ import annotations

import json
import unicodedata
from typing import Optional

try:
    from .value_clusterer import Cluster
    from .validation_layer import ValidationLayer
    from .llm_json_parser import parse_llm_json
except ImportError: 
    from value_clusterer import Cluster
    from validation_layer import ValidationLayer
    from llm_json_parser import parse_llm_json


# Prompt builder
def _build_cluster_prompt(
    column: str,
    ambiguous_clusters: list[Cluster],
    allowed_values: Optional[list[str]],
) -> str:
    groups = []
    for idx, cluster in enumerate(ambiguous_clusters):
        groups.append({
            "group_id": idx,
            "members": cluster.members,
            "suggested_canonical": cluster.canonical,
        })

    # ------------------------------------------------------------------
    # Two distinct prompt sections depending on mode
    # ------------------------------------------------------------------
    if allowed_values:
        allowed_section = (
            f"\nAllowed canonical values (you MUST pick from this list exactly):\n"
            f"{json.dumps(allowed_values, ensure_ascii=False)}\n"
        )
        canonical_rule = (
            "- The canonical value MUST be one of the allowed values, character-for-character."
        )
    else:
        allowed_section = (
            "\nNo predefined whitelist provided — you will discover the canonical form.\n"
        )
        canonical_rule = (
            "- Choose the most standard, complete, and commonly recognized form.\n"
            "- You MAY expand abbreviations to their full form (e.g., 'NY' → 'New York', "
            "'N.Y.' → 'New York', 'LA' → 'Los Angeles').\n"
            "- When expanding, use the most widely accepted full name.\n"
            "- If unsure whether to expand, prefer the most common member of the group."
        )

    return f"""You are resolving ambiguous value groups for a single dataset column.

Column: {column}{allowed_section}
Your task: for each group decide
  1. Whether all members genuinely refer to the same concept.
  2. What the canonical form should be.
  3. Your confidence (0.0–1.0).

STRICT RULES:
- If members refer to DIFFERENT concepts, set "reject": true and explain why.
{canonical_rule}
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


# Response parser (unchanged)
def _parse_llm_response(raw: str, n_groups: int) -> list[dict]:
    """Parse LLM JSON response, delegating fence-stripping to parse_llm_json."""
    try:
        parsed = parse_llm_json(raw, list)
    except ValueError:
        raise ValueError("No valid JSON array found in LLM response")

    by_id: dict[int, dict] = {}
    for item in parsed:
        gid = item.get("group_id")
        if gid is not None:
            by_id[int(gid)] = item

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


# Main resolver
class ClusterResolver:
    """
    Resolves clusters produced by :func:`cluster_column` into a concrete
    value mapping ``{original_value: canonical_value}``.

    Auto-resolved clusters are mapped directly.
    Ambiguous clusters trigger a single batched LLM call.
    All mappings pass through the ValidationLayer before acceptance.

    Parameters
    ----------
    chat_fn : callable
        Signature: ``(messages, completion_tokens) -> completion``
    validation : ValidationLayer
        Gates every mapping before acceptance.
    confidence_threshold : float
        Default threshold for strict mode. Discovery mode uses
        ``max(confidence_threshold, discovery_threshold)`` automatically.
    discovery_threshold : float
        Higher threshold used when ``allowed_values`` is not provided.
        Default 0.85 (more conservative to offset hallucination risk).
    """

    def __init__(
        self,
        chat_fn,
        validation: ValidationLayer,
        confidence_threshold: float = 0.7,
        discovery_threshold: float = 0.85,
    ):
        self._chat = chat_fn
        self.validation = validation
        self.threshold = confidence_threshold
        self.discovery_threshold = discovery_threshold

    def resolve(
        self,
        column: str,
        auto_clusters: list[Cluster],
        ambiguous_clusters: list[Cluster],
        allowed_values: Optional[list[str]] = None,  
    ) -> tuple[dict[str, str], list[dict]]:
        # stores accepted mappings (variant → canonical) and log entries
        mapping: dict[str, str] = {}
        log: list[dict] = []

        # Determine effective threshold based on mode (strict or discovery)
        is_discovery_mode = allowed_values is None
        effective_threshold = (
            max(self.threshold, self.discovery_threshold)
            if is_discovery_mode
            else self.threshold
        )

        # process clusters that can be resolved automatically (no LLM needed)
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
                    "mode": "strict" if not is_discovery_mode else "discovery",
                }
                if val_ok:
                    mapping[original] = canonical
                else:
                    entry["fallback_reason"] = f"validation failed: {val_reason}"
                log.append(entry)

        # --- Ambiguous clusters: one batched LLM call ---
        if ambiguous_clusters:
            llm_decisions = self._call_llm(column, ambiguous_clusters, allowed_values)
            
            # apply each LLM decision after confidence and validation checks
            for cluster, decision in zip(ambiguous_clusters, llm_decisions):
                self._apply_llm_decision(
                    column, cluster, decision, allowed_values,
                    mapping, log, effective_threshold
                )

        return mapping, log

    # LLM call
    def _call_llm(
        self,
        column: str,
        clusters: list[Cluster],
        allowed_values: Optional[list[str]],
    ) -> list[dict]:
        
        # build a prompt describing every ambiguous cluster
        prompt = _build_cluster_prompt(column, clusters, allowed_values)
        
        # send a single batched request to reduce LLm calls
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
            # parse the JSON response into one decision per cluster
            raw = self._chat(messages, completion_tokens=512)
            if not isinstance(raw, str):
                raw = raw.choices[0].message.content
            return _parse_llm_response(raw, len(clusters))
        except Exception as exc:
            # if LLM fails, reject every cluster and log the failure reason
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
        
    # Apply one LLM decision to the mapping + log
    def _apply_llm_decision(
        self,
        column: str,
        cluster: Cluster,
        decision: dict,
        allowed_values: Optional[list[str]],
        mapping: dict[str, str],
        log: list[dict],
        effective_threshold: float,
    ) -> None:
        
        # Extract and normalize the LLM's proposed canonical value.
        canonical_raw = decision.get("canonical") or ""
        canonical = unicodedata.normalize("NFKC", str(canonical_raw).strip())
        confidence = float(decision.get("confidence", 0.0))
        rejected = bool(decision.get("reject", False))
        llm_reason = str(decision.get("reason", ""))

        # determine if we are in discovery mode (no allowed_values provided)
        is_discovery_mode = allowed_values is None
        allowed_set = set(allowed_values) if allowed_values else set()

        mode_label = "discovery" if is_discovery_mode else "strict"

        # evaluate each value in the cluster independently 
        for original in cluster.members:

            # skip the canonical itself (no mapping needed)
            if original == canonical:
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
                    "mode": mode_label,
                })
                continue

            # 1. Rejection: rejected by LLM
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
                    "mode": mode_label,
                })
                continue

            # 2. In strict mode, canonical must be in allowed_values
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
                    "mode": mode_label,
                })
                continue

            # 3. Reject mappings below confidence threshold
            if confidence < effective_threshold:
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
                        f"confidence {confidence:.2f} < threshold {effective_threshold:.2f}"
                        f" (mode: {mode_label})"
                    ),
                    "validation_passed": False,
                    "validation_reason": "below confidence threshold",
                    "mode": mode_label,
                })
                continue

            # 4. Apply validation
            val_ok, val_reason = self.validation.validate(column, original, canonical)
            
            # 5. store accepted mapping
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
                "mode": mode_label,
            }
            log.append(entry)