from __future__ import annotations
from typing import Optional
import pandas as pd

from .detection_strategy import ColumnIssue, DetectionContext, DetectionResult, DetectionStrategy
from .llm_client import ChatMessage, LLMClient

try:
    from .cluster_resolver import ClusterResolver
    from .validation_layer import ValidationLayer
    from .value_clusterer import cluster_column, split_clusters
except ImportError:
    from cluster_resolver import ClusterResolver
    from validation_layer import ValidationLayer
    from value_clusterer import cluster_column, split_clusters


class CategoricalClusterDetector(DetectionStrategy):
    """
    Detects categorical inconsistencies via edit-distance clustering
    and optional LLM resolution for ambiguous cases.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        validation: ValidationLayer,
        confidence_threshold: float = 0.7,
        similarity_threshold: float = 0.35,
        max_unique_values: int = 500,
        min_unique_for_auto: int = 20,
        discovery_threshold: float = 0.8,
    ) -> None:
        self._llm_client = llm_client
        self._validation = validation
        self._confidence_threshold = confidence_threshold
        self._similarity_threshold = similarity_threshold
        self._max_unique_values = max_unique_values
        self._min_unique_for_auto = min_unique_for_auto

        # resolver decides the canonical value for each cluster,
        # using validation rules and LLM assistance if needed.
        self._resolver = ClusterResolver(
            chat_fn=self._chat_fn,
            validation=validation,
            confidence_threshold=confidence_threshold,
            discovery_threshold=discovery_threshold,
        )

    # ------------------------------------------------------------------
    # DetectionStrategy interface
    # ------------------------------------------------------------------
    def detect(
        self,
        series: pd.Series,
        context: DetectionContext,
    ) -> DetectionResult:
        allowed_values = context.allowed_values

        if self._should_skip(series.nunique(), allowed_values):
            print(
                f"  [{context.column}] skipped: {series.nunique()} unique values, "
                "no allowed_values registered"
            )
            return DetectionResult(issues=[], log=[])

        unique_values, value_counts = self._extract_unique_values(series, context.column)

        # Group similar categorical values using edit-distance similarity.
        clusters = cluster_column(
            unique_values=unique_values,
            value_counts=value_counts,
            allowed_values=allowed_values,  # Passes None in discovery mode
            similarity_threshold=self._similarity_threshold,
        )

        # Separate clusters that can be resolved automatically
        # from those requiring LLM assistance.
        auto_clusters, ambiguous_clusters = split_clusters(clusters)

        print(
            f"  [{context.column}] "
            f"clusters={len(clusters)}  "
            f"auto={len(auto_clusters)}  "
            f"ambiguous={len(ambiguous_clusters)}"
        )

        # Resolve each cluster into canonical values while applying
        # validation rules and confidence thresholds.
        mapping, validation_log = self._resolver.resolve(
            column=context.column,
            auto_clusters=auto_clusters,
            ambiguous_clusters=ambiguous_clusters,
            allowed_values=allowed_values,  # Passes None in discovery mode
        )

        # Convert accepted mappings into pipeline-standard issues
        # for the orchestrator to apply later.
        issues = [
            ColumnIssue(
                index=None,
                value=variant,
                confidence=self._confidence_from_log(variant, validation_log),
                reason=f"variant of '{canonical}'",
                kind="variant",
                canonical=canonical,
            )
            for variant, canonical in mapping.items()
        ]

        return DetectionResult(issues=issues, log=validation_log)

    def _should_skip(
        self,
        unique_count: int,
        allowed_values: Optional[list[str]],  # ← Type hint fixed
    ) -> bool:
        """
        Determine whether to skip a column entirely.

        - Strict mode (allowed_values provided): NEVER skip.
          User provided a whitelist — they want enforcement regardless of
          cardinality (e.g., ["Active", "active", "Inactive"] with 3 uniques).
          
        - Discovery mode (allowed_values is None): skip only if there's
          nothing to cluster (< 2 unique values). Low-cardinality columns
          like ["N.Y.", "NY", "New York"] MUST be processed to normalize
          abbreviations.
        """
        if allowed_values is not None:
            # Strict mode: always process
            return False
        else:
            # Discovery mode: need at least 2 values to form a cluster
            return unique_count < 2

    def _extract_unique_values(
        self,
        series: pd.Series,
        column: str,
    ) -> tuple[list[str], dict[str, int]]:
        vc = series.astype(str).value_counts()
        if len(vc) > self._max_unique_values:
            print(
                f"  [WARNING] '{column}' has {len(vc)} unique values — "
                f"truncating to {self._max_unique_values}."
            )
        unique_values = vc.index.tolist()[: self._max_unique_values]
        return unique_values, vc.to_dict()

    def _chat_fn(
        self,
        messages: list[dict],
        completion_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """
        Adapter: converts ClusterResolver's dict messages to ChatMessage,
        calls LLMClient, returns plain string content.
        """
        chat_messages = [
            ChatMessage(role=m["role"], content=m["content"])
            for m in messages
        ]
        response = self._llm_client.complete(
            messages=chat_messages,
            max_tokens=completion_tokens,
            temperature=temperature,
        )
        return response.content

    @staticmethod
    def _confidence_from_log(variant: str, log: list[dict]) -> float:
        for entry in log:
            if entry.get("original") == variant and entry.get("accepted"):
                return float(entry.get("confidence", 1.0))
        return 1.0